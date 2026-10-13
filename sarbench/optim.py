"""Optimizer groups for adapters, task heads, and optional unfrozen backbone weights."""


def parameter_groups(model, backbone_lr, head_lr, unfrozen_lr=None):
    """Keep heads in group 1, and optionally give original backbone weights their own LR."""
    backbone = [(name, p) for name, p in model.backbone.named_parameters() if p.requires_grad]
    heads = [p for name, p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
    if unfrozen_lr is None:
        return [{'params': [p for _, p in backbone], 'lr': backbone_lr, 'name': 'backbone'},
                {'params': heads, 'lr': head_lr, 'name': 'heads'}]
    adapters, original = [], []
    for name, parameter in backbone:
        target = adapters if any(part in name for part in ('.down.', '.up.', '.routers.')) else original
        target.append(parameter)
    return [{'params': adapters, 'lr': backbone_lr, 'name': 'adapters'},
            {'params': heads, 'lr': head_lr, 'name': 'heads'},
            {'params': original, 'lr': unfrozen_lr, 'name': 'unfrozen_backbone'}]
