import struct
import shutil
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def _collect(root, keep_dir, only_ext):
    root = Path(root)
    files, n_sample, n_miss = [], 0, 0
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample in sorted(p for p in cls.iterdir() if p.is_dir()):
            src = sample / keep_dir
            if not src.is_dir():
                n_miss += 1
                continue
            n_sample += 1
            for f in src.rglob("*"):
                if f.is_file() and (only_ext is None or f.suffix.lower() in only_ext):
                    files.append(str(f))
    print(f"样本文件夹 {n_sample} 个（缺 {keep_dir} 的 {n_miss} 个），待压文件 {len(files)} 个")
    return files


def _zip_chunk(job):
    idx, paths, root, tmp_dir, level = job
    root = Path(root)
    tmp = Path(tmp_dir) / f"part_{idx:05d}.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                         allowZip64=True, compresslevel=level) as zf:
        for p in paths:
            p = Path(p)
            zf.write(p, p.relative_to(root).as_posix())
    return str(tmp)


def _append_raw(zout, zin, info):
    zin.fp.seek(info.header_offset)
    head = zin.fp.read(30)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"本地头异常: {info.filename}")
    fn_len, ex_len = struct.unpack("<HH", head[26:30])
    zin.fp.seek(info.header_offset + 30 + fn_len + ex_len)
    data = zin.fp.read(info.compress_size)

    zi = zipfile.ZipInfo(info.filename, info.date_time)
    zi.compress_type = info.compress_type
    zi.create_system = info.create_system
    zi.external_attr = info.external_attr
    zi.flag_bits = info.flag_bits & ~0x08
    zi.CRC = info.CRC
    zi.compress_size = info.compress_size
    zi.file_size = info.file_size
    zi.header_offset = zout.fp.tell()

    zip64 = (zi.file_size > zipfile.ZIP64_LIMIT
             or zi.compress_size > zipfile.ZIP64_LIMIT)
    zout.fp.write(zi.FileHeader(zip64))
    zout.fp.write(data)
    zout.start_dir = zout.fp.tell()
    zout.filelist.append(zi)
    zout.NameToInfo[zi.filename] = zi
    zout._didModify = True


def build_zip(root_dir, output_zip, keep_dir="CONTEXT", only_ext=None,
              level=7, workers=24, chunks_per_worker=4, tmp_dir=None):
    root = Path(root_dir)
    out = Path(output_zip)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = _collect(root, keep_dir, only_ext)
    if not files:
        print("没有找到任何文件，检查 root_dir 和 keep_dir")
        return

    n_chunk = max(1, workers * chunks_per_worker)
    size = (len(files) + n_chunk - 1) // n_chunk
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ctx_parts_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    jobs = [(i, files[i * size:(i + 1) * size], str(root), str(tmp_root), level)
            for i in range(n_chunk) if files[i * size:(i + 1) * size]]

    print(f"并行压缩：{len(jobs)} 块 / {workers} 进程 / level {level}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_zip_chunk, jobs))

    print("合并中……")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for part in parts:
            with zipfile.ZipFile(part) as zin:
                for info in zin.infolist():
                    _append_raw(zout, zin, info)

    shutil.rmtree(tmp_root, ignore_errors=True)
    mb = out.stat().st_size / 1024 / 1024
    print(f"完成：{out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    build_zip(
        root_dir=r"Q:\context500\train",
        output_zip=r"C:\Users\YaMiSo\Documents\context500_train.zip",
        keep_dir="CONTEXT",
        only_ext={".csv"},
        level=7,
        workers=24,
        chunks_per_worker=4,
        tmp_dir=r"C:\Users\YaMiSo\Documents\_zip_tmp",
    )