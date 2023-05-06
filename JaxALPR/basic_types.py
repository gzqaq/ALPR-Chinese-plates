import jax
from typing import Union, List, Tuple, Dict, Any

KeyArray = Union[jax.Array, jax._src.prng.PRNGKeyArray]

DatasetItem = Tuple[str, List[List[int]], List[int]]
DatasetList = List[DatasetItem]

BatchType = Tuple[jax.Array, jax.Array]

MetricType = Dict[str, List[float]]
