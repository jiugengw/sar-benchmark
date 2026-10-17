# Invalid branch comparison

This initial comparison evaluated the same branch twice: Ultralytics backend
setup overrides the model's `end2end` flag using its `nms` argument. These scores
must not be used as evidence that the branches are equivalent.

The exporter and screening script now set `nms` explicitly for each branch. Valid
full-holdout branch comparisons are saved under
`results/deadline-yolo26m-329644/e5_spatial_head/`, and the exporter asserts that
the backend selected the requested branch.
