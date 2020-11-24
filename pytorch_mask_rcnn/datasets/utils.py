from .voc_dataset import VOCDataset

__all__ = ["datasets"]


def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ["voc", "coco"]
    if ds == choice[0]:
        return VOCDataset(*args, **kwargs)
    else:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))