#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import os
import pickle
import re
import typing

from . import classes, utils

"""
This module provides functions for reading and processing image data, including parsing filenames to extract relevant metadata and generating input dictionaries.
"""


def _read_images(
    dirpaths: str | list[str],
    input_names: tuple[str, ...] | None = None,
    all_possible_fluorophores: list[str] | None = None,
    digital_inputs: bool = True,
    input_base_unit: str = "",
    filename_pattern: str | None = None,
    original_image_format: str = "lof",
    bit_depth: int = 2**8,
    num_x_sections: int = 8,
    num_y_sections: int = 8,
    validity_determining_channels: tuple[int] = (0, ),
    crop: bool = False,
    filename_filter: typing.Callable[[str], bool] | None = None,
    cache_dir: str = "./cache",
):
	"""
	Read and process images from specified directories.

	Args:
		dirpaths (str | list[str]): Path(s) to directories containing image files.
		input_names (tuple[str, ...], optional): Names of inputs to extract from filenames. Must be provided if :any:`filename_pattern` includes inputs specifiers.
		all_possible_fluorophores (list[str], optional): List of potential fluorophores (currently unused). Must be provided if :any:`filename_pattern` includes fluorophore specifiers.
		digital_inputs (bool, optional): Whether inputs are digital (True/False). Defaults to True.
		input_base_unit (str, optional): Base unit for non-digital inputs. Must be provided if :any:`filename_pattern` includes non-digital input specifiers (e.g. :literal:`0.1_mM_EtOH`). Defaults to an empty string.
		filename_pattern (str, optional): Regex pattern for parsing filenames. Optional, defaults to match :literal:`without_Estr_0.1_mM_EtOH_24_hours_after_exposure_plant_1_leaf_1_image_1.npz`.
		original_image_format (:literal:`lof` | :literal:`czi`, optional): Format of the original images. Defaults to :literal:`lof`.
		bit_depth (int, optional): Bit depth of the images. Defaults to :math:`2^8`.
		num_x_sections (int, optional): Number of sections along the x-axis. Defaults to 8.
		num_y_sections (int, optional): Number of sections along the y-axis. Defaults to 8.
		validity_determining_channels (tuple[int], optional): Channels used for determining validity. Defaults to (0, ).
		crop (bool, optional): Whether cropping is enabled. Defaults to False.
		filename_filter (typing.Callable[[str], bool], optional): Function to filter filenames. If provided, the callable will be applied independently to each filename, and filenames returning :literal:`False` will be skipped.
		cache_dir (str, optional): Directory for caching processed images. Defaults to :literal:`./cache`.

	Yields:
		classes.UnsectionedConfocalImage: Processed image data as UnsectionedConfocalImage objects.
	"""
	if filename_pattern is None:
		assert input_names is not None, "input_names must be provided if filename_pattern is not specified"

		filename_pattern = rf"(?P<input_1_value>.+)_{input_names[0]}_(?P<input_2_value>.+)_{input_names[1]}_(?P<hours_after_exposure>\d+)_hours_after_exposure_plant_(?P<plant_num>\d+)_leaf_(?P<leaf_num>\d+)_image_(?P<position_num>\d+).npz"

	if isinstance(dirpaths, str):
		dirpaths = [dirpaths]

	for dirpath in dirpaths:
		filenames = [filename for dirpath in dirpaths for filename in os.listdir(dirpath)]

		for filename in filenames:
			tokenized_filename = re.match(filename_pattern, filename)

			if tokenized_filename is None:
				continue

			if filename_filter is not None:
				if filename_filter(filename) == False:
					continue

			filename_without_extension = os.path.splitext(filename)[0]

			pickle_filename = os.path.join(cache_dir, filename_without_extension, f"{filename_without_extension}.pkl")

			if os.path.exists(pickle_filename):
				with open(pickle_filename, "rb") as f:
					image_data = pickle.load(f)

				yield image_data
			else:
				inputs = _generate_inputs_dict_from_regex_match(
				    tokenized_filename,
				    input_names,
				    digital_inputs,
				    input_base_unit,
				)

				fluorophores = None  # TODO: Implement logic to extract fluorophores from filename, similar to inputs

				hours_after_exposure = int(tokenized_filename.groupdict()["hours_after_exposure"])

				kwargs = {}
				for group_name in ["plant_num", "leaf_num", "position_num"]:
					kwargs[group_name] = None

					if group_name in tokenized_filename.groupdict().keys():
						kwargs[group_name] = int(tokenized_filename.groupdict()[group_name])

				yield classes.UnsectionedConfocalImage(
				    filename = os.path.join(dirpath, filename),
				    inputs = inputs,
				    fluorophores = fluorophores,
				    hours_after_exposure = hours_after_exposure,
				    num_x_sections = num_x_sections,
				    num_y_sections = num_y_sections,
				    original_image_format = original_image_format,
				    bit_depth = bit_depth,
				    validity_determining_channels = validity_determining_channels,
				    crop = crop,
				    kwargs = kwargs,
				)


def read_images(
    dirpaths: str | list[str],
    input_names: tuple[str, ...] | None = None,
    all_possible_fluorophores: list[str] | None = None,
    digital_inputs: bool = True,
    input_base_unit: str = "",
    filename_pattern: str | None = None,
    original_image_format: str = "lof",
    bit_depth: int = 2**8,
    num_x_sections: int = 8,
    num_y_sections: int = 8,
    validity_determining_channels: tuple[int] = (0, ),
    crop: bool = False,
    filename_filter: typing.Callable[[str], bool] | None = None,
    cache_dir: str = "./cache",
):
	if filename_pattern is None:
		assert input_names is not None, "input_names must be provided if filename_pattern is not specified"

		filename_pattern = rf"(?P<input_1_value>.+)_{input_names[0]}_(?P<input_2_value>.+)_{input_names[1]}_(?P<hours_after_exposure>\d+)_hours_after_exposure_plant_(?P<plant_num>\d+)_leaf_(?P<leaf_num>\d+)_image_(?P<position_num>\d+).npz"

	if isinstance(dirpaths, str):
		dirpaths = [dirpaths]

	num_files = len([
	    filename for dirpath in dirpaths for filename in os.listdir(dirpath)
	    if re.match(filename_pattern, filename) and (filename_filter is None or filename_filter(filename) == True)
	])

	return (
	    _read_images(
	        dirpaths,
	        input_names,
	        all_possible_fluorophores,
	        digital_inputs,
	        input_base_unit,
	        filename_pattern,
	        original_image_format,
	        bit_depth,
	        num_x_sections,
	        num_y_sections,
	        validity_determining_channels,
	        crop,
	        filename_filter,
	        cache_dir,
	    ),
	    num_files,
	)


def _generate_inputs_dict_from_regex_match(
    tokenized_filename: re.Match[str],
    input_names: tuple[str, ...] | None,
    digital_inputs: bool,
    input_base_unit: str,
) -> dict[str, bool] | dict[str, tuple[int | float, str]] | None:
	"""
	Generate a dictionary of inputs based on a regex match.

	Args:
		tokenized_filename (re.Match[str]): Regex match object containing parsed filename components.
		input_names (tuple[str, ...] | None): Names of inputs to extract from the filename components.
		digital_inputs (bool): Whether inputs are digital (True/False).
		input_base_unit (str): Base unit for non-digital input values.

	Returns:
		dict[str, bool] | dict[str, tuple[int | float, str]] | None: A dictionary of inputs extracted from the filename.
	"""
	if input_names is None:
		return

	inputs = {}

	for input_num, input_name in enumerate(input_names):
		input_value = tokenized_filename.groupdict()[f"input_{input_num + 1}_value"]

		if digital_inputs:
			if input_value == "with":
				inputs[input_name] = True
			elif input_value == "without":
				inputs[input_name] = False
			else:
				raise ValueError(f"Invalid digital input value {input_value} for input {input_name}")
		else:
			inputs[input_name] = utils.get_standardized_units(input_value, input_base_unit)

	return inputs
