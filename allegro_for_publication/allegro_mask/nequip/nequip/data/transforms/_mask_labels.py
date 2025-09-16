from typing import Optional, Union, List

import torch

from nequip.data import AtomicData, AtomicDataDict, _NODE_FIELDS, _LONG_FIELDS


NaN: float = float("nan")


class MaskByAtomTag:
    """Mask out a per-atom label according to the value of some atom 'tag' field.

    Example usage in YAML file:

        dataset_pre_transform: !!python/object:nequip.data.transforms.MaskByAtomTag {'tag_values_to_mask': [0], 'fields_to_mask': ['forces']}

    See `examples/mask_labels/minimal_mask.yaml` for details and a runable example.
    """

    tag_field: str = AtomicDataDict.ATOM_TAG_KEY
    tag_values_to_mask: Optional[List[int]] = None
    fields_to_mask: List[str]

    def __init__(
        self,
        tag_values_to_mask: int,
        fields_to_mask: List[str],
        tag_field: str = AtomicDataDict.ATOM_TAG_KEY,
    ):
        self.tag_values_to_mask = tag_values_to_mask
        self.tag_field = tag_field
        self.fields_to_mask = fields_to_mask

    def __call__(
        self, data: Union[AtomicDataDict.Type, AtomicData]
    ) -> Union[AtomicDataDict.Type, AtomicData]:
        assert self.tag_values_to_mask is not None, "tag_value_to_mask is required"
        assert self.tag_field in _NODE_FIELDS and self.tag_field in _LONG_FIELDS
        mask = torch.isin(
            data[self.tag_field],
            torch.as_tensor(self.tag_values_to_mask, dtype=torch.long),
        ).squeeze(-1)
        for f in self.fields_to_mask:
            assert f in _NODE_FIELDS
            data[f][mask] = NaN
        return data
