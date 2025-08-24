#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import concurrent.futures
import gc
import math
import operator
import os
import pickle
import re
import signal
import threading
import time
import typing

import matplotlib.colors as mplcol  # type: ignore[import-untyped]
import numpy as np
import PIL.Image  # type: ignore[import-untyped]
import tqdm  # type: ignore[import-untyped]

if typing.TYPE_CHECKING:
	import genlogic.optical.classes as classes  # type: ignore[import-untyped]

_CHANNEL_COLOUR_MAPPING = {
    0: "red",
    1: "green",
    2: "orange",
    "*": "yellow",
}


def channel_to_colour(channel: int) -> str:
	global _CHANNEL_COLOUR_MAPPING

	if channel in _CHANNEL_COLOUR_MAPPING:
		return _CHANNEL_COLOUR_MAPPING[channel]
	else:
		return _CHANNEL_COLOUR_MAPPING["*"]


def unique(list_of_items):
	seen = []
	unique_items = []
	for item in list_of_items:
		if item not in seen:
			seen.append(item)
			unique_items.append(item)
	return unique_items


# From https://stackoverflow.com/a/35360388
def debug_pickle(instance):
	attribute = None

	for k, v in instance.__dict__.items():
		try:
			pickle.dumps(v)
		except:
			attribute = k
			break

	return attribute


def recursively_create_directories(directories: dict) -> None:
	for value in directories.values():
		if isinstance(value, str):
			if os.path.exists(value):
				continue
			os.makedirs(value, exist_ok = True)
		elif isinstance(value, dict):
			recursively_create_directories(value)


def _recursively_update_dict(
    target_dict: dict[str, typing.Any],
    source_dict: dict[str, typing.Any],
) -> None:
	for key, value in source_dict.items():
		if isinstance(value, dict):
			_recursively_update_dict(target_dict[key], value)
		else:
			target_dict[key] = value


def get_standardized_units(
    raw_string: str,
    base_unit: str,
) -> tuple[float, str]:
	raw_string = raw_string.replace(" ", "")
	raw_string = raw_string.replace("_", "")

	tokenized_filename = re.search(rf"(?P<value>[\.\d]+)(?P<prefix>[p|n|u|m|k|M]{{0,1}})(?P<unit>{base_unit})",
	                               raw_string)

	assert tokenized_filename is not None, f"Unexpected raw string: {raw_string}"

	value = float(tokenized_filename.group("value"))
	prefix = tokenized_filename.group("prefix")
	base_unit = tokenized_filename.group("unit")

	multiplier = {
	    "p": 1e-12,
	    "n": 1e-9,
	    "u": 1e-6,
	    "m": 1e-3,
	    "": 1,
	    "k": 1e3,
	    "M": 1e6,
	}

	standardized_value = (value * multiplier[prefix], base_unit)

	return standardized_value


def get_humanreadable_units(standardized_value: tuple[float, str], ) -> tuple[float, str]:
	lowest_thousands_power = math.floor(math.log(standardized_value[0], 1000)) if standardized_value[0] != 0 else 0

	prefix = {
	    -4: "p",
	    -3: "n",
	    -2: "u",
	    -1: "m",
	    0: "",
	    1: "k",
	    2: "M",
	}

	return (standardized_value[0] / 1000**lowest_thousands_power,
	        prefix[lowest_thousands_power] + standardized_value[1])


def _process_terminator(parent_pid):
	pid = os.getpid()

	def f():
		while True:
			try:
				os.kill(parent_pid, 0)
			except OSError:
				os.kill(pid, signal.SIGTERM)
			else:
				time.sleep(1)

	thread = threading.Thread(target = f, daemon = True)
	thread.start()


def update_thresholds_for_sections(
    sections: list["classes.ConfocalImageSection"],
    new_thresholds: dict,
    repickle: bool = False,
    concurrency_strategy: typing.Literal["thread", "process"] = "thread",
    chunksize: int = 8,
    max_workers: int = 8,
):

	if concurrency_strategy == "thread":
		Executor = concurrent.futures.ThreadPoolExecutor
	elif concurrency_strategy == "process":
		Executor = concurrent.futures.ProcessPoolExecutor
		repickle = True

	with tqdm.tqdm(
	    total = len(sections),
	    desc = "Updating thresholds for sections",
	    unit = "section",
	    leave = True,
	) as pbar:
		with Executor(max_workers = max_workers) as executor:

			futures = executor.map(
			    operator.methodcaller(
			        "update_thresholds",
			        new_thresholds = new_thresholds,
			        repickle = repickle,
			    ),
			    sections,
			    chunksize = chunksize,
			)

			for _ in futures:
				pbar.update(1)

	if concurrency_strategy == "process":
		reload_sections_from_pickle(sections)


def reload_sections_from_pickle(sections: list["classes.ConfocalImageSection"]):
	for section_idx in range(len(sections)):
		section = sections[section_idx]

		pickle_filename = section.cache_filepaths["pickle"]

		with open(pickle_filename, "rb") as f:
			sections[section_idx] = pickle.load(f)


def select_sections(
    images: list["classes.UnsectionedConfocalImage"],
    num_images: int,
    force_reselect: bool = False,
    dump_pickle: bool = True,
) -> list["classes.ConfocalImageSection"]:
	sections = []

	if not force_reselect and os.path.exists("selected_sections.pkl"):
		tqdm.tqdm.write("Loading sections from pickle file...")
		with open("selected_sections.pkl", "rb") as f:
			sections = pickle.load(f)
		return sections

	with tqdm.tqdm(
	    total = num_images,
	    desc = "Selecting sections",
	    unit = "image",
	    leave = True,
	) as pbar:
		with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as executor:
			futures = executor.map(_get_sections_from_image, images, chunksize = 1)

			for future in futures:
				sections.extend(future)
				pbar.update(1)

		if dump_pickle:
			with open("selected_sections.pkl", "wb") as f:
				pickle.dump(sections, f, protocol = pickle.HIGHEST_PROTOCOL)

	return sections


def _get_sections_from_image(image):
	sections = list(image.sections)

	del image

	gc.collect()

	return sections


def vibrance_colourmap(base_colour: str, num_colours: int = 2**8):
	"""
	Create a colourmap with a vibrance gradient.

	Args:
		base_colour (str): The base colour of the colourmap. Must be a valid matplotlib colour name.
		num_colours (int): The number of colours in the colourmap. Defaults to 2**8.

	Returns:
		mpl.colors.LinearSegmentedColormap: The colourmap.
	"""
	colours = ["black", base_colour]
	colour_map = mplcol.LinearSegmentedColormap.from_list("vibrance", colours, num_colours)

	return colour_map


def get_perspective_image(
        slice_as_matrix,
        upsampling_factor: int = 1,
        interpolation_method = PIL.Image.BICUBIC,
        border_width: int = 2,
        border_colour: tuple[int, int, int, int] = (60, 60, 60, 255),
        cmap: mplcol.LinearSegmentedColormap = vibrance_colourmap("white", num_colours = 2**8),
) -> PIL.Image:

	side_length = slice_as_matrix.shape[0]
	upsampled_image = PIL.Image.fromarray(slice_as_matrix.astype("uint8")).resize(
	    (side_length * upsampling_factor, side_length * upsampling_factor),
	    interpolation_method,
	)
	upsampled_image_as_matrix = np.array(upsampled_image)
	upsampled_side_length = upsampled_image_as_matrix.shape[0]

	# create border by shrinking the content area
	border_pixels = int(border_width * np.sqrt(2))  # adjust for 45-degree rotation

	image_with_border = np.full_like(upsampled_image_as_matrix, 0)  # Start with black/empty

	# place the image in the centre
	if border_pixels < upsampled_side_length // 2:
		image_with_border[border_pixels:-border_pixels, border_pixels:-border_pixels] = \
                                          upsampled_image_as_matrix[border_pixels:-border_pixels, border_pixels:-border_pixels]

	mask_as_matrix = np.zeros_like(image_with_border)
	mask_as_matrix[border_pixels:-border_pixels, border_pixels:-border_pixels] = 255

	border_mask = np.ones_like(image_with_border) * 255
	border_mask[border_pixels:-border_pixels, border_pixels:-border_pixels] = 0

	S1 = np.array([
	    [np.sqrt(2), 0, 0],
	    [0, np.sqrt(2), 0],
	    [0, 0, 1],
	])

	T1 = np.array([
	    [1, 0, upsampled_side_length / (2 * np.sqrt(2))],
	    [0, 1, upsampled_side_length / (2 * np.sqrt(2))],
	    [0, 0, 1],
	])

	R1 = np.array([
	    [np.cos(np.pi / 4), np.sin(np.pi / 4), 0],
	    [-np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
	    [0, 0, 1],
	])

	T2 = np.array([
	    [1, 0, -upsampled_side_length / 2],
	    [0, 1, -upsampled_side_length / 2],
	    [0, 0, 1],
	])

	y_scale = 1.5

	S2 = np.array([
	    [1, 0, 0],
	    [0, y_scale, 0],
	    [0, 0, 1],
	])

	T3 = np.array([
	    [1, 0, 0],
	    [0, 1, -(upsampled_side_length - upsampled_side_length / y_scale)],
	    [0, 0, 1],
	])

	transformation_matrix = (S1 @ T1 @ R1 @ T2 @ S2 @ T3).flatten()

	slice_image = PIL.Image.fromarray(image_with_border.astype("uint8"))
	slice_image_transformed = slice_image.transform(
	    size = (upsampled_side_length, upsampled_side_length),
	    method = PIL.Image.AFFINE,
	    data = transformation_matrix,
	)

	mask_image = PIL.Image.fromarray(mask_as_matrix.astype("uint8"))
	mask_image_transformed = mask_image.transform(
	    size = (upsampled_side_length, upsampled_side_length),
	    method = PIL.Image.AFFINE,
	    data = transformation_matrix,
	)

	border_mask_image = PIL.Image.fromarray(border_mask.astype("uint8"))
	border_mask_transformed = border_mask_image.transform(
	    size = (upsampled_side_length, upsampled_side_length),
	    method = PIL.Image.AFFINE,
	    data = transformation_matrix,
	)

	slice_array = np.array(slice_image_transformed)
	slice_array_normalized = slice_array / (2**8 - 1)
	slice_array_coloured = cmap(slice_array_normalized)
	slice_array_coloured = (slice_array_coloured * (2**8 - 1)).astype("uint8")

	content_mask_array = np.array(mask_image_transformed)
	border_mask_array = np.array(border_mask_transformed)

	slice_image_rgba = slice_array_coloured.copy()

	slice_image_rgba[:, :, 3] = content_mask_array

	border_areas = (border_mask_array > 0) & (content_mask_array == 0)
	slice_image_rgba[border_areas] = border_colour

	return PIL.Image.fromarray(slice_image_rgba, mode = "RGBA")


def generate_dirs():
	"""
	Generate directories for storing results.

	Args:
		None

	Returns:
		tuple: The directories.
	"""
	for root_dir in [
	    ".",
	    "./for_paper",
	]:
		for image_type in ["sections", "unsectioned_images"]:
			for export_type in [
			    "maximum_intensity_projections",
			    "slicewise",
			    "image_stacks",
			]:
				dir_path = os.path.join(root_dir, "exports", export_type, image_type)
				os.makedirs(dir_path, exist_ok = True)

			for graph_type in [
			    "validation/distributions",
			]:
				dir_path = os.path.join(root_dir, "graphs", graph_type, image_type)

		for graph_type in [
		    "linegraphs",
		    "violins",
		]:
			dir_path = os.path.join(root_dir, "graphs", graph_type)
			os.makedirs(dir_path, exist_ok = True)
