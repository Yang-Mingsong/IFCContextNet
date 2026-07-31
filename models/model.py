import torch.nn as nn

from models.encoder import RGCNEncoder, GCNEncoder, RGATSemEncoder, SemAvailabilityEncoder
from models.fusion import ConcatFusion

VALID_MODALITIES = ('rep', 'sem', 'spa')



class IFCComponentClassifier(nn.Module):

    def __init__(self, modalities, in_dim, hidden_dim, out_dim, num_classes, sem_availability_only,
                 rep_metagraph=None, sem_metagraph=None, n_hops=2):
        super(IFCComponentClassifier, self).__init__()

        self.modalities = [m for m in VALID_MODALITIES if m in set(modalities)]

        self.encoders = nn.ModuleDict()
        if 'rep' in self.modalities:
            if rep_metagraph is None:
                raise ValueError("rep 模态需要 rep_metagraph")
            self.encoders['rep'] = RGCNEncoder(in_dim, hidden_dim, out_dim, rep_metagraph)
        if 'spa' in self.modalities:
            self.encoders['spa'] = GCNEncoder(in_dim, hidden_dim, out_dim, n_hops)
        if 'sem' in self.modalities:
            if sem_availability_only:
                self.encoders['sem'] = SemAvailabilityEncoder(out_dim)
            else:
                if sem_metagraph is None:
                    raise ValueError("sem 模态需要 sem_metagraph")
                self.encoders['sem'] = RGATSemEncoder(in_dim, hidden_dim, out_dim, sem_metagraph)

        self.fusion = ConcatFusion(in_dims=[out_dim] * len(self.modalities),
                                   hidden_dim=out_dim, num_classes=num_classes)

    def forward(self, batch):
        feats = [self.encoders[m](batch[m]) for m in self.modalities]
        return self.fusion(feats)