import os
import sys
import csv
import gc
import hashlib
import subprocess
import multiprocessing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty as _QueueEmpty

import numpy as np
import pandas as pd

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ifc_elem

from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound


# ====
# 配置
# ====
# 语义上下文中排除的关系(属性集/材质/空间结构包含/类型定义)
EXCLUDED_RELATIONSHIPS = {
    "IfcRelDefinesByProperties", "IfcRelAssociatesMaterial",
    "IfcRelContainedInSpatialStructure", "IfcRelDeclares", "IfcRelDefinesByType",
}

# 不参与几何上下文的类型
SKIP_TYPES = {
    "IfcOpeningElement", "IfcSpace", "IfcAnnotation", "IfcGrid", "IfcVirtualElement",
    "IfcSite", "IfcProject", "IfcBuilding", "IfcBuildingStorey",
}

# 已知会让 create_shape 卡死/极慢的类型:绝不生成自身几何(可按需往里加)
RISKY_TYPES = set()

# 按 GUID 黑名单(实例级)
RISKY_GUIDS = set()
RISKY_GUIDS.add("1De98_NMv5fOuvlRh7qiw5")

USE_AABB_WORKER = True   # True:子进程 worker(自动跳过卡死元素);False:主进程内(易卡死)
PER_ELEM_TIMEOUT = 300    # 单元素 AABB 超时(秒)。

# 邻接判据 = 放大后的 OBB 是否重叠;两档膨胀给出分级边类型:
OBB_GAP_CONTACT = 0.02   # 小膨胀，近似真接触,吸收三角化/建模抖动
OBB_GAP_NEAR = 0.10      # 大膨胀，容差内相邻,吸收建模缝隙
PROXIMITY_MAX_M = 2.0    # 救援
FORCED_MAX_M = 5.0

MAX_HOPS = 5             # 空间上下文最大跳数
NODE_BUDGET = 200        # 单个上下文图最多收纳多少节点(防稠密模型里整栋楼被拉进来)

SUCCESS_MARKER, FAILED_MARKER, TIMEOUT_MARKER, ZERO_MARKER = (
    "_SUCCESS", "_FAILED", "_TIMEOUT", "_ZERO_NEIGHBOR",
)


# =============================================
# 几何后端:形状 / OBB / 中心 / 间距(单位米,带缓存)
# =============================================
class GeometryStore:

    def __init__(self, model):
        self.model = model
        self._settings = self._make_settings(disable_opening=False)
        self._settings_fast = self._make_settings(disable_opening=True)
        self._tri_settings = ifcopenshell.geom.settings()
        self._tri_settings.set(self._tri_settings.USE_WORLD_COORDS, True)
        self._shape_cache = {}
        self._obb_c_cache = {}
        self._obb_n_cache = {}
        self._aabb_cache = {}

    @staticmethod
    def _make_settings(disable_opening):
        s = ifcopenshell.geom.settings()
        s.set(s.USE_WORLD_COORDS, True)
        s.set(s.USE_PYTHON_OPENCASCADE, True)
        if disable_opening:
            s.set(s.DISABLE_OPENING_SUBTRACTIONS, True)
        return s

    def _create_raw(self, elem):
        """仅从构件自身 Representation 生成形状;失败返回 None。"""
        if not getattr(elem, "Representation", None):
            return None
        for settings in (self._settings, self._settings_fast):
            try:
                shape = ifcopenshell.geom.create_shape(settings, elem).geometry
                if shape is not None and not (hasattr(shape, "IsNull") and shape.IsNull()):
                    return shape
            except Exception:
                continue
        return None

    def _compound_from_decomposition(self, elem):
        """聚合体回退:把分解链子构件的形状拼成一个 compound。"""
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        count = 0
        for sub in (ifc_elem.get_decomposition(elem) or []):
            if sub.is_a() in SKIP_TYPES:
                continue
            sub_shape = self._create_raw(sub)
            if sub_shape is not None:
                builder.Add(compound, sub_shape)
                count += 1
        return compound if count > 0 else None

    def _fast_verts(self, elem):
        """用 ifcopenshell 快速三角化(不走 OCC)取顶点 (N,3) 米;失败返回 None。"""
        if not getattr(elem, "Representation", None):
            return None
        try:
            tri = ifcopenshell.geom.create_shape(self._tri_settings, elem)
            v = np.asarray(tri.geometry.verts, dtype=np.float64).reshape(-1, 3)
            return v if len(v) else None
        except Exception:
            return None

    def aabb(self, guid):
        """轴对齐包围盒 (min_xyz, max_xyz)(米);无几何返回 None。"""
        if guid in self._aabb_cache:
            return self._aabb_cache[guid]
        elem = self.model.by_guid(guid)
        box = None
        if (elem is not None and elem.is_a() not in SKIP_TYPES
                and elem.is_a() not in RISKY_TYPES and guid not in RISKY_GUIDS):
            verts = self._fast_verts(elem)
            if verts is None:                      # 聚合体回退:并所有子件顶点
                parts = []
                for sub in (ifc_elem.get_decomposition(elem) or []):
                    if sub.is_a() in SKIP_TYPES:
                        continue
                    v = self._fast_verts(sub)
                    if v is not None:
                        parts.append(v)
                if parts:
                    verts = np.vstack(parts)
            if verts is not None and len(verts):
                box = (verts.min(axis=0), verts.max(axis=0))
        self._aabb_cache[guid] = box
        return box

    def shape(self, guid):
        """返回构件形状;无几何返回 None。"""
        if guid in self._shape_cache:
            return self._shape_cache[guid]
        elem = self.model.by_guid(guid)
        shape = None
        if (elem is not None and elem.is_a() not in SKIP_TYPES
                and elem.is_a() not in RISKY_TYPES and guid not in RISKY_GUIDS):
            shape = self._create_raw(elem) or self._compound_from_decomposition(elem)
        self._shape_cache[guid] = shape
        return shape

    def _obb_at(self, guid, gap, cache):
        """在指定膨胀量下取 OBB(已缓存)。useTriangulation=False:直接从几何算,无需网格化。"""
        if guid in cache:
            return cache[guid]
        shape = self.shape(guid)
        obb = None
        if shape is not None:
            try:
                obb = Bnd_OBB()
                brepbndlib.AddOBB(shape, obb, False, True, False)
                obb.Enlarge(gap)         # 放大不改变中心
            except Exception:
                obb = None
        cache[guid] = obb
        return obb

    def obb_contact(self, guid):
        return self._obb_at(guid, OBB_GAP_CONTACT, self._obb_c_cache)

    def obb_near(self, guid):
        return self._obb_at(guid, OBB_GAP_NEAR, self._obb_n_cache)


def adjacency_type(geom, guid_a, guid_b):
    """返回相邻类型"""
    na, nb = geom.obb_near(guid_a), geom.obb_near(guid_b)
    if na is None or nb is None or na.IsOut(nb):
        return None
    ca, cb = geom.obb_contact(guid_a), geom.obb_contact(guid_b)
    if ca is not None and cb is not None and not ca.IsOut(cb):
        return "CONTACT"
    return "NEAR"


# =================
# AABB 子进程 worker
# =================
def _aabb_worker_proc(ifc_path, req_q, resp_q):

    try:
        import ifcopenshell as _ifc
        import ifcopenshell.geom as _ifc_geom
        import ifcopenshell.util.element as _ifc_elem
        import numpy as _np
        model = _ifc.open(ifc_path)
        settings = _ifc_geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        def _verts(elem):
            """自身 Representation 快速三角化取顶点;失败返回 None。"""
            if not getattr(elem, "Representation", None):
                return None
            try:
                tri = _ifc_geom.create_shape(settings, elem)
                v = _np.asarray(tri.geometry.verts, dtype=_np.float64).reshape(-1, 3)
                return v if len(v) else None
            except Exception:
                return None

        resp_q.put("READY")
        while True:
            guid = req_q.get()
            if guid is None:
                break
            try:
                elem = model.by_guid(guid)
                if elem is None:
                    resp_q.put(None)
                    continue
                v = _verts(elem)
                if v is None:                          # 聚合体回退:并所有子件顶点
                    parts = []
                    for sub in (_ifc_elem.get_decomposition(elem) or []):
                        if sub.is_a() in SKIP_TYPES:
                            continue
                        sv = _verts(sub)
                        if sv is not None:
                            parts.append(sv)
                    if parts:
                        v = _np.vstack(parts)
                if v is None:
                    resp_q.put(None)
                    continue
                resp_q.put((v.min(axis=0).tolist(), v.max(axis=0).tolist()))
            except Exception:
                resp_q.put(None)
    except Exception as e:
        try:
            resp_q.put(("FATAL", str(e)))
        except Exception:
            pass


class _AabbWorker:

    def __init__(self, ifc_path, per_elem_timeout=PER_ELEM_TIMEOUT):
        self.ifc_path = ifc_path
        self.timeout = per_elem_timeout
        self.hung_guids = set()
        self._spawn()

    def _spawn(self):
        self.req_q = multiprocessing.Queue()
        self.resp_q = multiprocessing.Queue()
        self.proc = multiprocessing.Process(
            target=_aabb_worker_proc,
            args=(self.ifc_path, self.req_q, self.resp_q),
            daemon=True,
        )
        self.proc.start()
        try:
            ready = self.resp_q.get(timeout=600)   # 加载大 IFC 给足 10 分钟
        except _QueueEmpty:
            raise RuntimeError("aabb worker 启动超时(IFC 加载未完成)")
        if ready != "READY":
            raise RuntimeError(f"aabb worker 启动失败:{ready}")

    def aabb(self, guid):
        """返回 (min_arr, max_arr) 或 None;卡死时自动登记 guid 并杀子进程重生,返回 None。"""
        if guid in self.hung_guids:
            return None
        try:
            self.req_q.put(guid)
            r = self.resp_q.get(timeout=self.timeout)
        except _QueueEmpty:
            print(f"        [HANG] {guid} 超时 {self.timeout}s -> 自动跳过、重启子进程",
                  flush=True)
            self.hung_guids.add(guid)
            try: self.proc.kill()
            except Exception: pass
            try: self.proc.join(timeout=5)
            except Exception: pass
            self._spawn()
            return None
        if r is None or (isinstance(r, tuple) and len(r) >= 1 and r[0] == "FATAL"):
            return None
        mn, mx = r
        return (np.asarray(mn, dtype=np.float64), np.asarray(mx, dtype=np.float64))

    def close(self):
        try: self.req_q.put(None)
        except Exception: pass
        try: self.proc.join(timeout=10)
        except Exception: pass
        if self.proc.is_alive():
            try: self.proc.kill()
            except Exception: pass


# =======
# 空间索引
# =======
class SpatialIndex:

    def __init__(self, model, geom, ifc_path=None):
        self.geom = geom
        self.guids = []
        mins, maxs = [], []
        elements = [e for e in model.by_type("IfcElement") if e.is_a() not in SKIP_TYPES]
        total = len(elements)
        print(f"        index over {total} elements ...", flush=True)

        # 选取 AABB 来源:子进程 worker(可中断卡死的 C 调用) vs 主进程内(易卡死)
        worker = None
        if USE_AABB_WORKER and ifc_path is not None:
            worker = _AabbWorker(ifc_path, per_elem_timeout=PER_ELEM_TIMEOUT)
            aabb_fn = lambda g: worker.aabb(g) if g not in RISKY_GUIDS else None
        else:
            aabb_fn = geom.aabb

        try:
            for i, elem in enumerate(elements, 1):
                box = aabb_fn(elem.GlobalId)
                if box is not None:
                    self.guids.append(elem.GlobalId)
                    mins.append(box[0])
                    maxs.append(box[1])

                print(f"        ...indexing {i}/{total}  构件 {elem.GlobalId}", flush=True)
        finally:
            if worker is not None:
                worker.close()
                if worker.hung_guids:
                    print(f"        [auto-skip] 本模型自动跳过 {len(worker.hung_guids)} "
                          f"个卡死元素(已加入运行时 RISKY_GUIDS):", flush=True)
                    for g in sorted(worker.hung_guids):
                        print(f"           - {g}", flush=True)
                    RISKY_GUIDS.update(worker.hung_guids)   # 让本模型后续 shape() 也跳过

        self.mins = np.asarray(mins, dtype=np.float64) if mins else np.empty((0, 3))
        self.maxs = np.asarray(maxs, dtype=np.float64) if maxs else np.empty((0, 3))
        self._centers = (self.mins + self.maxs) * 0.5 if len(self.guids) else np.empty((0, 3))
        self._row = {g: i for i, g in enumerate(self.guids)}
        print(f"        index done: {len(self.guids)} elements", flush=True)

    def candidates(self, guid, gap=OBB_GAP_NEAR):
        """返回 AABB(各放大 gap)与 guid 的 AABB 重叠的所有构件(不含自身)。"""
        i = self._row.get(guid)
        if i is None or len(self.guids) == 0:
            return []
        tmin = self.mins[i] - gap
        tmax = self.maxs[i] + gap
        # 轴对齐盒重叠:每个轴上区间都相交
        mask = np.all((self.maxs >= tmin) & (self.mins <= tmax), axis=1)
        mask[i] = False
        return [self.guids[j] for j in np.nonzero(mask)[0]]

    def nearest(self, guid, excluded):
        """按 AABB 中心距返回最近的一个构件(救援用);返回 (guid, dist)。"""
        i = self._row.get(guid)
        if i is None or len(self.guids) == 0:
            return None, float("inf")
        d = np.linalg.norm(self._centers - self._centers[i], axis=1)
        d[i] = np.inf
        for g in excluded:
            j = self._row.get(g)
            if j is not None:
                d[j] = np.inf
        k = int(np.argmin(d))
        return (self.guids[k], float(d[k])) if np.isfinite(d[k]) else (None, float("inf"))


# =================================
# 空间上下文图:以目标为中心的多跳邻接图
# =================================
def build_spatial_graph(target_guid, target_class, geom, index, excluded):

    nodes = [("class_node", 0, target_class)]
    local_id = {target_guid: 0}

    def node_id(guid):
        if guid not in local_id:
            local_id[guid] = len(nodes)
            nodes.append(("class_node", local_id[guid], geom.model.by_guid(guid).is_a()))
        return local_id[guid]

    edges = []
    checked = set()
    visited = {target_guid} | set(excluded)
    frontier = [target_guid]

    for hop in range(1, MAX_HOPS + 1):
        next_frontier = []
        for src in frontier:
            for cand in index.candidates(src):
                if cand in excluded:
                    continue
                pair = (src, cand) if src < cand else (cand, src)
                if pair in checked:
                    continue
                checked.add(pair)

                rel = adjacency_type(geom, src, cand)
                if rel is not None:
                    edges.append(("class_node", local_id[src], "class_node", node_id(cand), rel, hop))
                    if cand not in visited and len(nodes) < NODE_BUDGET:
                        visited.add(cand)
                        next_frontier.append(cand)

        # 仅 hop-1、目标一个相邻构件都没有时,保证目标有空间上下文
        if hop == 1 and not edges:
            best, dist = index.nearest(target_guid, visited)
            if best is not None and dist <= FORCED_MAX_M:
                rel = "PROXIMITY" if dist < PROXIMITY_MAX_M else "FORCED"
                edges.append(("class_node", 0, "class_node", node_id(best), rel, 1))
                visited.add(best)
                next_frontier.append(best)

        frontier = next_frontier
        if not frontier:
            break

    return nodes, edges


def decomposition_family(model, target):
    """目标的分解家族(后代 + 祖先)GUID 集合,从空间邻居中剔除。"""
    family = {target.GlobalId}
    for sub in (ifc_elem.get_decomposition(target) or []):
        family.add(sub.GlobalId)
    parent = ifc_elem.get_aggregate(target)
    while parent is not None:
        family.add(parent.GlobalId)
        parent = ifc_elem.get_aggregate(parent)
    return family


# =====================
# 语义上下文图(IFC 关系)
# =====================
class SemanticExtractor:

    _attr_is_collection_cache = {}

    def __init__(self, schema="IFC4"):
        self.schema = schema
        self.nodes = []
        self.edges = []
        self._id_of_entity = {}
        self._counter = {"class_node": 1, "attribute_node": 0, "type_node": 0}

    def _is_collection_attr(self, entity_type, attr_name):
        key = (entity_type, attr_name)
        if key in self._attr_is_collection_cache:
            return self._attr_is_collection_cache[key]
        result = False
        try:
            decl = ifcopenshell.schema_by_name(self.schema).declaration_by_name(entity_type)
            for attr in decl.all_attributes():
                if attr.name() == attr_name:
                    type_str = str(attr.type_of_attribute()).lower()
                    result = any(kw in type_str for kw in ("set", "list", "array", "bag"))
                    break
        except Exception:
            pass
        self._attr_is_collection_cache[key] = result
        return result

    def _class_node(self, entity, is_target=False):
        if entity.id() in self._id_of_entity:
            return self._id_of_entity[entity.id()]
        nid = 0 if is_target else self._counter["class_node"]
        if not is_target:
            self._counter["class_node"] += 1
        self.nodes.append(("class_node", nid, entity.is_a()))
        self._id_of_entity[entity.id()] = nid
        return nid

    def _attribute_node(self, name):
        nid = self._counter["attribute_node"]
        self._counter["attribute_node"] += 1
        self.nodes.append(("attribute_node", nid, name))
        return nid

    def _tuple_node(self):
        nid = self._counter["type_node"]
        self._counter["type_node"] += 1
        self.nodes.append(("type_node", nid, "TUPLE"))
        return nid

    def extract(self, target, model):
        root = self._class_node(target, is_target=True)
        for rel in (model.get_inverse(target) or []):
            if not rel.is_a("IfcRelationship") or rel.is_a() in EXCLUDED_RELATIONSHIPS:
                continue
            rel_id = self._class_node(rel)

            for attr_name, attr_val in rel.get_info().items():
                if attr_val is None or attr_name in (
                    "OwnerHistory", "GlobalId", "Name", "Description", "id", "type"
                ):
                    continue
                contains_target, others = self._split_participants(attr_val, target)
                if not contains_target and not others:
                    continue

                attr_id = self._attribute_node(attr_name)
                self.edges.append(("class_node", rel_id, "attribute_node", attr_id, "hasAttribute"))

                if self._is_collection_attr(rel.is_a(), attr_name):
                    tuple_id = self._tuple_node()
                    self.edges.append(("attribute_node", attr_id, "type_node", tuple_id, "hasValue"))
                    if contains_target:
                        self.edges.append(("type_node", tuple_id, "class_node", root, "hasValue"))
                    for o in others:
                        self.edges.append(("type_node", tuple_id, "class_node",
                                           self._class_node(o), "hasValue"))
                else:
                    if contains_target:
                        self.edges.append(("attribute_node", attr_id, "class_node", root, "hasValue"))
                    for o in others:
                        self.edges.append(("attribute_node", attr_id, "class_node",
                                           self._class_node(o), "hasValue"))

    @staticmethod
    def _split_participants(attr_val, target):
        contains_target, others = False, []
        values = attr_val if isinstance(attr_val, (list, tuple)) else [attr_val]
        for v in values:
            if isinstance(v, ifcopenshell.entity_instance):
                if v.id() == target.id():
                    contains_target = True
                elif v.id() != 0:
                    others.append(v)
        return contains_target, others


# ===============
# 单构件处理 + IO
# ===============
def process_component(guid, fname, model, geom, index, output_root):
    category = fname.split("_")[1] if len(fname.split("_")) >= 2 else "Uncategorized"
    save_dir = os.path.join(output_root, category, fname, "CONTEXT")
    succ = os.path.join(save_dir, SUCCESS_MARKER)
    if os.path.exists(succ):
        return "skip"

    os.makedirs(save_dir, exist_ok=True)
    fail, tout, zero = (os.path.join(save_dir, m) for m in (FAILED_MARKER, TIMEOUT_MARKER, ZERO_MARKER))

    try:
        target = model.by_guid(guid)
        if target is None:
            raise ValueError(f"GUID {guid} not found")

        sem = SemanticExtractor()
        sem.extract(target, model)

        family = decomposition_family(model, target)
        nodes, edges = build_spatial_graph(guid, target.is_a(), geom, index, family)
        if len(edges) == 0:
            open(zero, "w").write("no spatial neighbor")
        elif os.path.exists(zero):
            os.remove(zero)            # 上轮零邻居、本轮有边:清掉旧标记

        _write_csv(os.path.join(save_dir, "semantics_node.csv"), sem.nodes)
        _write_csv(os.path.join(save_dir, "semantics_edge.csv"), sem.edges)
        _write_csv(os.path.join(save_dir, "geometry_node.csv"), nodes)
        _write_csv(os.path.join(save_dir, "geometry_edge.csv"), edges)

        for m in (fail, tout):
            if os.path.exists(m):
                os.remove(m)
        open(succ, "w").write("done")
        return "ok"

    except Exception as e:
        open(fail, "w").write(str(e))
        return f"fail:{e}"


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def process_model(ifc_path, task_list, output_root):
    """一个完整模型的 worker:建一次几何后端/索引,处理其下全部目标构件。"""
    name = os.path.basename(ifc_path)
    print(f"[open]  {name} ({len(task_list)} components)", flush=True)
    try:
        model = ifcopenshell.open(ifc_path)
    except Exception as e:
        return f"OPEN FAILED {name}: {e}"

    geom = GeometryStore(model)
    print(f"[index] {name}: building spatial index ...", flush=True)
    index = SpatialIndex(model, geom, ifc_path=ifc_path)   # 启用 AABB worker
    print(f"[index] {name}: {len(index.guids)} elements indexed", flush=True)

    stats = defaultdict(int)
    total = len(task_list)
    for i, (guid, fname) in enumerate(task_list, 1):
        result = process_component(guid, fname, model, geom, index, output_root)
        stats[result.split(":")[0]] += 1
        print(f"   [{i}/{total}] {fname} -> {result}", flush=True)

    del model
    gc.collect()
    return (f"{name} | ok:{stats['ok']} "
            f"skip:{stats['skip']} fail:{stats['fail']}")


# ================
# 任务扫描 + 主入口
# ================
def scan_tasks(input_csv, output_dir, quiet=False):
    """按完整模型分组待处理任务;已成功且结果非空的跳过。"""
    df = pd.read_csv(input_csv)
    tasks = defaultdict(list)
    done = 0
    for _, r in df.iterrows():
        ifc_path = str(r["Mapped_IFC_Path"]).strip().replace('"', "")
        guid = str(r["Extracted_GUID"]).strip()
        fname = os.path.splitext(os.path.basename(str(r["Train_File"]).strip()))[0] or guid
        category = fname.split("_")[1] if len(fname.split("_")) >= 2 else "Uncategorized"

        edge_csv = os.path.join(output_dir, category, fname, "CONTEXT", "geometry_edge.csv")
        succ = os.path.join(output_dir, category, fname, "CONTEXT", SUCCESS_MARKER)
        if os.path.exists(succ) and os.path.exists(edge_csv):
            done += 1
            continue
        tasks[ifc_path].append((guid, fname))

    pending = sum(len(v) for v in tasks.values())
    if not quiet:
        print(f"[scan] 已完成 {done} | 待处理 {pending}(模型数 {len(tasks)})")
    return tasks


def reset_zero_neighbor(output_dir):
    n = 0
    for dirpath, _, filenames in os.walk(output_dir):
        if ZERO_MARKER in filenames and SUCCESS_MARKER in filenames:
            try:
                os.remove(os.path.join(dirpath, SUCCESS_MARKER))
                n += 1
            except OSError:
                pass
    print(f"[reset] 已重置 {n} 个零邻居构件为待处理", flush=True)
    return n


def main(input_csv, output_dir, workers=16, serial=False, maxtasksperchild=None):
    tasks = scan_tasks(input_csv, output_dir)
    if not tasks:
        print("所有任务已完成。")
        return

    jobs = [(path, sub, output_dir) for path, sub in tasks.items()]
    if serial:
        for job in jobs:
            print(process_model(*job))
    else:
        with multiprocessing.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
            for res in pool.starmap(process_model, jobs, chunksize=1):
                print(res)


# ===================================================
# 进程级隔离运行(本地段错误 0xC0000005 / 卡死,可断点续跑)
# ===================================================

def _crashed_marker_path(output_dir, ifc_path):
    h = hashlib.md5(ifc_path.encode("utf-8")).hexdigest()[:12]
    d = os.path.join(output_dir, "_crashed_models")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{h}.crashed")


def _worker_one_model(ifc_path, input_csv, output_dir):
    """子进程入口:只处理这一个模型的待办构件。"""
    tasks = scan_tasks(input_csv, output_dir, quiet=True)
    sub = tasks.get(ifc_path)
    if sub:
        print(process_model(ifc_path, sub, output_dir), flush=True)


def main_isolated(input_csv, output_dir, workers=8, timeout=1800,
                  retry_crashed=False, retry_zero_neighbor=False):
    if retry_zero_neighbor:
        reset_zero_neighbor(output_dir)

    crashed_dir = os.path.join(output_dir, "_crashed_models")
    if retry_crashed and os.path.isdir(crashed_dir):
        removed = 0
        for fn in os.listdir(crashed_dir):
            try:
                os.remove(os.path.join(crashed_dir, fn))
                removed += 1
            except OSError:
                pass
        print(f"[isolated] retry_crashed:已清除 {removed} 个崩溃标记,将重试这些模型", flush=True)

    tasks = scan_tasks(input_csv, output_dir)
    paths = [p for p in tasks if not os.path.exists(_crashed_marker_path(output_dir, p))]
    skipped = len(tasks) - len(paths)
    if skipped:
        print(f"[isolated] 跳过已知崩溃模型 {skipped} 个", flush=True)
    if not paths:
        print("所有任务已完成(或剩余均为崩溃模型)。")
        return

    def run_one(path):
        name = os.path.basename(path)
        try:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", path, input_csv, output_dir],
                timeout=timeout,
            )
            if r.returncode == 0:
                return f"OK       {name}"
            open(_crashed_marker_path(output_dir, path), "w").write(str(r.returncode))
            return f"CRASH({r.returncode}) {name}  -> 已标记跳过"
        except subprocess.TimeoutExpired:
            open(_crashed_marker_path(output_dir, path), "w").write("timeout")
            return f"TIMEOUT  {name}  -> 已标记跳过"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(run_one, paths):
            print(res, flush=True)


if __name__ == "__main__":
    # 子进程分支:由 main_isolated 以 [--worker <ifc> <csv> <out>] 调起
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _worker_one_model(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main_isolated(
            input_csv=r"D:\PythonProject\PythonProject\BIMCompNet_5000_train.csv",
            output_dir=r"D:\Context5000\train",
            workers=24,            # 同时跑几个模型子进程;按 CPU/内存调
            timeout=1000000,       # 单模型超时(秒),卡死即杀
            retry_crashed=True,    # True:清除崩溃标记,重试曾崩溃的模型(重试完可改回 False)
            retry_zero_neighbor=True,  # 本次修复后第一次重跑设 True,跑完改回 False
        )