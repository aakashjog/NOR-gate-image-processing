#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import abc
import functools
import itertools
import json
import os
import pickle
import typing
import warnings
import zipfile

import czifile  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import PIL.Image  # type: ignore[import-untyped]

import genlogic.optical.utils as utils


class Inputs():
	"""
	A class for managing and formatting sorted inputs.

	Attributes:
		inputs_dict (dict["input name", bool | tuple["value", "unit"]] | None):
			Dictionary containing names and values of inputs (as booleans of number-unit tuples). E.g.: {"input1": True, "input2": (5, "nM")}. Defaults to None.
	"""

	def __init__(self, inputs_dict: dict[str, bool] | dict[str, tuple[int | float, str]] | None = None):
		"""
		Initialize the Inputs object.

		Args:
			inputs_dict (dict[str, bool] | dict[str, tuple[int | float, str]], optional):
				Dictionary containing names and values of inputs (as booleans or number-unit tuples). E.g.: {"input1": True, "input2": (5, "nM")}. Optional, defaults to None.
		"""
		if inputs_dict is not None:
			self.inputs_dict = {key: inputs_dict[key] for key in sorted(inputs_dict.keys())}
		else:
			self.inputs_dict = None

	@property
	def input_names(self) -> list[str]:
		"""
		The names of the inputs.

		Args:
			None

		Returns:
			list[str]: List of input names, sorted alphabetically.
		"""
		if self.inputs_dict is None:
			return []

		return list(self.inputs_dict.keys())

	def __hash__(self) -> int:
		if self.inputs_dict is None:
			return 0

		return hash(tuple(sorted(self.inputs_dict.items())))

	@property
	def input_values(self) -> list[bool | tuple[int | float, str]]:
		"""
		The values of the inputs.

		Args:
			None

		Returns:
			list[bool | tuple[int | float, str]]: List of input values, sorted alphabetically by input names.
		"""

		if self.inputs_dict is None:
			return []

		return list(self.inputs_dict.values())

	def __repr__(self) -> str:
		return f"Inputs({self.inputs_dict})"

	def __str__(self) -> str:
		return f"Inputs: {self.pretty_str(format = 'generic')}"

	def pretty_str(self, format = "generic") -> str:
		"""
		Get a pretty string representation of the inputs.

		Args:
			format (str): Format of the pretty string. Options are "generic", "legend", and "text". Defaults to "generic".
		Returns:
			str: Pretty string representation of the inputs.
		"""

		if self.inputs_dict is None:
			return ""

		pretty_string = ""
		pretty_string_sections = []

		match format:
			case "generic":
				for input_name, input_value in self.inputs_dict.items():
					if isinstance(input_value, bool):
						pretty_string_sections.append(f"{input_name} = {input_value}")
					elif isinstance(input_value, tuple) and len(input_value) == 2:
						pretty_string_sections.append(f"{input_name} = {input_value[0]} {input_value[1]}")
					else:
						raise ValueError(f"Unsupported input value type: {type(input_value)} for input {input_name}")

				pretty_string = ", ".join(pretty_string_sections)
			case "legend":
				print(f"getting legend")
				for input_name, input_value in self.inputs_dict.items():
					if isinstance(input_value, bool):
						pretty_string_sections.append(f"{input_name}{'$+$' if input_value == True else '$-$'}")
					elif isinstance(input_value, tuple) and len(input_value) == 2:
						pretty_string_sections.append(f"{input_value[0]} {input_value[1]} {input_name}")
					else:
						raise ValueError(f"Unsupported input value type: {type(input_value)} for input {input_name}")

				pretty_string = "/".join(pretty_string_sections)
			case "text":
				print(f"getting text")
				for input_name, input_value in self.inputs_dict.items():
					if isinstance(input_value, bool):
						pretty_string_sections.append(f"{'with' if input_value == True else 'without'} {input_name}")
					elif isinstance(input_value, tuple) and len(input_value) == 2:
						pretty_string_sections.append(f"{input_value[0]} {input_value[1]} {input_name}")
					else:
						raise ValueError(f"Unsupported input value type: {type(input_value)} for input {input_name}")

				pretty_string = ", ".join(pretty_string_sections)
			case _:
				raise ValueError(f"Unsupported format: {format}")

		return pretty_string

	def __eq__(self, other) -> bool:
		if not isinstance(other, self.__class__):
			return False

		if self.inputs_dict is None and other.inputs_dict is None:
			return True
		elif self.inputs_dict is None or other.inputs_dict is None:
			return False

		if len(self.inputs_dict) != len(other.inputs_dict):
			return False

		for key in self.inputs_dict.keys():
			if key not in other.inputs_dict or self.inputs_dict[key] != other.inputs_dict[key]:
				return False

		return True

	def __lt__(self, other) -> bool:
		if not isinstance(other, self.__class__):
			return False

		if self.inputs_dict is None and other.inputs_dict is None:
			return True
		elif self.inputs_dict is None or other.inputs_dict is None:
			return False

		return self.input_values < other.input_values

	def __gt__(self, other) -> bool:
		if not isinstance(other, self.__class__):
			return False

		if self.inputs_dict is None and other.inputs_dict is None:
			return False
		elif self.inputs_dict is None or other.inputs_dict is None:
			return True

		return self.input_values > other.input_values

	def __le__(self, other) -> bool:
		return self < other or self == other

	def __ge__(self, other) -> bool:
		return self > other or self == other


class Fluorophores():
	"""
	A class for managing and formatting fluorophores.

	Attributes:
		fluorophores (list[str] | None):
			List of fluorophores present in an image. Defaults to None.
	"""

	def __init__(
	    self,
	    fluorophores: list[str] | None = None,
	):
		"""
		Initialize the Fluorophores object.

		Args:
			fluorophores (list[str], optional):
				List of fluorophores present in an image. Optional, defaults to None.
		"""
		if fluorophores is not None:
			self.fluorophores = [fluorophore for fluorophore in sorted(fluorophores)]
		else:
			self.fluorophores = None

	def pretty_str(
	    self,
	    exclude_fluorophores: list[str] = [],
	    exclude_indices: list[int] = [],
	    format: str = "generic",
	) -> str:
		"""
		Get a pretty string representation of the fluorophores.

		Args:
			format (str): Format of the pretty string. Currently only "generic" is supported. Defaults to "generic".
		Returns:
			str: Pretty string representation of the inputs.
		"""

		if self.fluorophores is None:
			return ""

		pretty_string = ""
		pretty_string_sections = []

		match format:
			case _:
				for fluorophore_idx, fluorophore in enumerate(self.fluorophores):
					if fluorophore_idx not in exclude_indices and fluorophore not in exclude_fluorophores:
						pretty_string_sections.append(fluorophore)

				pretty_string = ", ".join(pretty_string_sections)

		return pretty_string


class Image():
	"""
	A class for managing image processing operations.

	Attributes:
		filename (str): The name of the image file.
		filename_without_ext (str): The filename without its extension.
		filepath (str): Relative path to the image file.
		filename_extension (str): File extension of the image file.
		shape (tuple): Shape of the image represented as a numpy array.
		crop (bool): Indicates whether the image is cropped.
		crop_x (int | None): X-coordinate for cropping.
		crop_y (int | None): Y-coordinate for cropping.
		crop_width (int | None): Width of the cropped region.
		crop_height (int | None): Height of the cropped region.
		bit_depth (int): Bit depth of the image.
	"""

	def __init__(
	    self,
	    filename: str,
	    crop: bool = False,
	    crop_x: int | None = None,
	    crop_y: int | None = None,
	    crop_width: int | None = None,
	    crop_height: int | None = None,
	    bit_depth: int = 2**8,
	    shape: tuple[int, ...] | np.ndarray | None = None,
	):
		"""
		Initialize an Image object.

		Args:
			filename (str): Path to the image file.
			crop (bool): Whether to crop the image.
			crop_x (int | None): X-coordinate for cropping.
			crop_y (int | None): Y-coordinate for cropping.
			crop_width (int | None): Width of the cropped region.
			crop_height (int | None): Height of the cropped region.
			bit_depth (int): Bit depth of the image.
		"""
		self.filename = os.path.basename(filename)
		self.filename_without_ext = os.path.splitext(self.filename)[0]
		self.filepath = os.path.relpath(os.path.join(os.path.dirname(filename), os.path.basename(filename)))
		self.filename_extension = os.path.splitext(self.filename)[1][1:].lower()

		if shape is not None:
			self.shape = shape
		else:
			self.shape = self._image.shape

		self.crop = crop

		if crop_x is None:
			crop_x = self.shape[3] // 2
		if crop_y is None:
			crop_y = self.shape[2] // 2
		if crop_width is None:
			crop_width = self.shape[3] // 3 * 2
		if crop_height is None:
			crop_height = self.shape[2] // 3 * 2

		self.crop_x = crop_x
		self.crop_y = crop_y
		self.crop_width = crop_width
		self.crop_height = crop_height

		self.bit_depth = bit_depth

	@functools.cached_property
	def num_channels(self) -> int:
		return self._image.shape[0]

	@functools.cached_property
	def num_slices(self) -> int:
		return self._image.shape[1]

	@functools.cached_property
	def _image(self) -> np.ndarray:
		match self.filename_extension:
			case "czi":
				czi_file = czifile.CziFile(self.filepath)
				return czi_file.asarray().squeeze()
			case "lof":
				return np.load(f"{self.filepath.replace('.lof', '')}.npz")["arr_0"]
			case "npz":
				return np.load(self.filepath)["arr_0"]
			case _:
				raise ValueError(f"Unsupported image format: {self.filename_extension}")

	@functools.cached_property
	def _cropped_image(self) -> np.ndarray:
		if self.crop:
			assert self.crop_x is not None
			assert self.crop_y is not None
			assert self.crop_width is not None
			assert self.crop_height is not None

			return self._image[:, :, self.crop_x - self.crop_width // 2:self.crop_x + self.crop_width // 2,
			                   self.crop_y - self.crop_height // 2:self.crop_y + self.crop_height // 2]
		else:
			return self._image

	def get_image(self, crop: bool | None = None, slice_nums: int | list[int] | None = None) -> np.ndarray:
		if crop is None:
			crop = self.crop

		if crop and self._cropped_image is not None:
			image = self._cropped_image
		else:
			image = self._image

		if slice_nums is not None:
			return image[:, slice_nums, :, :]
		else:
			return image

	def get_max_intensity_projection_image(self, crop: bool | None = None) -> np.ndarray:
		if crop is None:
			crop = self.crop

		if crop and self._cropped_image is not None:
			image = self._cropped_image
		else:
			image = self._image

		return np.max(image, axis = 1)

	def _write_image_with_PIL(
	    self,
	    image: np.ndarray,
	    export_filename: str,
	    export_format: str = "png",
	    cmap: str = "gray",
	):
		image_for_export = PIL.Image.fromarray(image.astype(np.uint8))

		plt.imsave(
		    export_filename,
		    image_for_export,
		    cmap = cmap,
		    vmin = 0,
		    vmax = 2**8 - 1,
		    format = export_format,
		)

	def _export(
	    self,
	    export_base_filename: str,
	    export_filename_extension: str = "png",
	    channels: list[int] | None = None,
	    slice_nums: list[int] | None = None,
	    crop: bool | None = None,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	    max_intensity_projection: bool = False,
	):
		if crop is None:
			crop = self.crop

		if channels is None:
			channels = list(range(self.num_channels))

		if slice_nums is None:
			slice_nums = list(range(self.num_slices))

		if max_intensity_projection:
			image = self.get_max_intensity_projection_image(crop = crop)
			slice_nums = [0]  # max intensity projection only has one slice
		else:
			image = self.get_image(crop = crop)

		if self.bit_depth != 2**8:
			image = image / self.bit_depth * 2**8

		image = np.clip(image * colour_scaling_factor, 0, 2**8 - 1)

		if max_intensity_projection:
			assert len(
			    image.shape
			) == 3, f"Max intensity projection image must have 3 dimensions (channels, height, width), got {len(image.shape)} ({image.shape})"

			for channel_num in channels:
				export_filename = f"{export_base_filename}_{self.shape[3]}x{self.shape[2]}_channel_{channel_num}"

				if self.crop:
					export_filename = f"{export_filename}_cropped"

				export_filename = f"{export_filename}.{export_filename_extension}"

				self._write_image_with_PIL(
				    image = image[channel_num, :, :],
				    export_filename = export_filename,
				    export_format = export_filename_extension,
				    cmap = utils.vibrance_colourmap(utils.channel_to_colour(channel_num), num_colours = 2**8),
				)
		else:
			for channel_num in channels:
				for slice_num in slice_nums:
					export_filename = f"{export_base_filename}_{self.shape[3]}x{self.shape[2]}_channel_{channel_num}_slice_{slice_num}"

					if self.crop:
						export_filename = f"{export_filename}_cropped"

					export_filename = f"{export_filename}.{export_filename_extension}"

					self._write_image_with_PIL(
					    image = image[channel_num, slice_num, :, :],
					    export_filename = export_filename,
					    export_format = export_filename_extension,
					    cmap = utils.vibrance_colourmap(utils.channel_to_colour(channel_num), num_colours = 2**8),
					)

		if clear_cache:
			del self._image

	def export_slices(
	    self,
	    export_base_filename: str,
	    export_filename_extension: str = "png",
	    channels: list[int] | None = None,
	    slice_nums: list[int] | None = None,
	    crop: bool | None = None,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	):
		self._export(
		    export_base_filename = export_base_filename,
		    export_filename_extension = export_filename_extension,
		    channels = channels,
		    slice_nums = slice_nums,
		    crop = crop,
		    clear_cache = clear_cache,
		    colour_scaling_factor = colour_scaling_factor,
		    max_intensity_projection = False,
		)

	def export_max_intensity_projection(
	    self,
	    export_base_filename: str,
	    export_filename_extension: str = "png",
	    channels: list[int] | None = None,
	    crop: bool | None = None,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	):
		self._export(
		    export_base_filename = export_base_filename,
		    export_filename_extension = export_filename_extension,
		    channels = channels,
		    crop = crop,
		    clear_cache = clear_cache,
		    colour_scaling_factor = colour_scaling_factor,
		    max_intensity_projection = True,
		)

	def export_image_stack(
	        self,
	        export_base_filename: str,
	        export_filename_extension: str = "png",
	        channels: list[int] | None = None,
	        slice_nums: list[int] | None = None,
	        crop: bool | None = None,
	        clear_cache: bool = False,
	        colour_scaling_factor: float | int = 1,
	        v_offset_factor: int = 4,
	        image_alpha: float | int = 1,
	        upsampling_factor: int = 1,
	        border_width: int = 2,
	        border_colour: tuple[int, int, int, int] = (60, 60, 60, 255),
	):
		"""
		Export an image stack as a series of images with perspective transformation.

		Args:
			export_base_filename (str): Base filename for the exported images.
			export_filename_extension (str): File extension for the exported images.
			channels (list[int] | None): List of channel numbers to export. If None, all channels are exported.
			slice_nums (list[int] | None): List of slice numbers to export. If None, all slices are exported.
			crop (bool | None): Whether to crop the image before exporting. If None, uses the object's crop attribute.
			clear_cache (bool): Whether to clear the cached image after exporting.
			colour_scaling_factor (float | int): Factor to scale the colour values of the image.
			v_offset_factor (int): Vertical offset factor for stacking slices.
			image_alpha (float | int): Alpha value for the image layers.
			upsampling_factor (int): Factor by which to upsample the image.
			border_width (int): Width of the border around each slice in the stack.
			border_colour (tuple[int, int, int, int]): RGBA colour of the border.

		Returns:
			None
		"""

		if crop is None:
			crop = self.crop

		if channels is None:
			channels = list(range(self.num_channels))

		if slice_nums is None:
			slice_nums = list(range(self.num_slices))

		image = self.get_image(crop = crop)
		image = np.clip(image * colour_scaling_factor, 0, 2**8 - 1)

		v_offset = int(image.shape[2] * upsampling_factor / v_offset_factor)

		for channel_num in channels:
			cmap = utils.vibrance_colourmap(
			    utils.channel_to_colour(channel_num),
			    num_colours = 2**8,
			)

			transformed_slice_images = []

			for slice_num in slice_nums:
				transformed_slice_image = utils.get_perspective_image(
				    slice_as_matrix = image[channel_num, slice_num, :, :],
				    upsampling_factor = upsampling_factor,
				    border_width = border_width,
				    border_colour = border_colour,
				    cmap = cmap,
				)

				transformed_slice_images.append(transformed_slice_image)

			num_stack_layers = len(transformed_slice_images)
			total_v_offset = (num_stack_layers - 1) * v_offset
			canvas_width = image.shape[2] * upsampling_factor + 2 * border_width
			canvas_height = (image.shape[2] * upsampling_factor + 2 * border_width) + total_v_offset

			canvas = np.zeros((canvas_height, canvas_width, 4), dtype = np.float32)

			for slice_idx, transformed_slice_image in enumerate(transformed_slice_images):
				layer_idx = num_stack_layers - 1 - slice_idx
				y_offset = layer_idx * v_offset

				transformed_slice_image_array_RGBA = np.array(transformed_slice_image).astype(np.float32) / (2**8 - 1)
				alpha_channel = transformed_slice_image_array_RGBA[:, :, 3] * image_alpha

				y_end = min(y_offset + transformed_slice_image_array_RGBA.shape[0], canvas_height)
				x_end = min(transformed_slice_image_array_RGBA.shape[1], canvas_width)

				canvas_region = canvas[y_offset:y_end, 0:x_end, :]
				layer_region = transformed_slice_image_array_RGBA[0:y_end - y_offset, 0:x_end, :]
				layer_alpha = alpha_channel[0:y_end - y_offset, 0:x_end, np.newaxis]

				canvas[y_offset:y_end, 0:x_end, :3] = (layer_alpha * layer_region[:, :, :3] +
				                                       (1 - layer_alpha) * canvas_region[:, :, :3])

				canvas[y_offset:y_end, 0:x_end, 3] = np.maximum(canvas_region[:, :, 3], layer_alpha.squeeze())

			export_filename = f"{export_base_filename}_{self.shape[3]}x{self.shape[2]}_channel_{channel_num}"

			if self.crop:
				export_filename = f"{export_filename}_cropped"

			export_filename = f"{export_filename}.{export_filename_extension}"

			fig, ax = plt.subplots(figsize = (canvas_width / 300, canvas_height / 300), dpi = 300)

			ax.set_xlim(0, canvas_width)
			ax.set_ylim(0, canvas_height)
			ax.axis("off")
			ax.set_position([0, 0, 1, 1])
			ax.imshow(
			    canvas,
			    extent = (0, canvas_width, 0, canvas_height),
			    origin = "upper",
			    interpolation = "nearest",
			)

			fig.savefig(
			    export_filename,
			    format = export_filename_extension,
			    dpi = 300,
			    bbox_inches = "tight",
			    pad_inches = 0,
			    transparent = True,
			    facecolor = "none",
			)

			plt.close(fig)

		if clear_cache:
			del self._image


class ValidityIntensityMetrics():
	"""
	A class for calculating validity and intensity metrics for an image.

	Attributes:
		default_thresholds (dict): Default thresholds for validity and calculation metrics.
		validities_filename_suffix (str): Suffix for validities filename.
		mean_intensities_of_all_slices_filename_suffix (str): Suffix for mean intensities of all slices filename.
		mean_intensities_of_valid_slices_filename_suffix (str): Suffix for mean intensities of valid slices filename.
		mean_intensities_of_max_intensity_projection_filename_suffix (str): Suffix for mean intensities of max intensity projection filename.
		parent_image (Image): The image for which metrics are calculated.
		image_x_size (int): X dimension of the image.
		image_y_size (int): Y dimension of the image.
		image_z_size (int): Z dimension (number of slices) of the image.
		image_num_channels (int): Number of channels in the image.
		validity_determining_channels (tuple[int]): Channels used to determine validity.
		_thresholds (dict): Thresholds used for metrics calculations.
		_validities (np.ndarray | None): Array indicating validity of slices.
		_mean_intensities (dict[str, np.ndarray | None]): Mean intensities for slices and max intensity projection.
	"""
	default_thresholds = {
	    "validity": {
	        "slice": {
	            "minimum_pixel_intensity": 2,
	            "minimum_valid_pixels_factor": 0.2,
	            "_minimum_valid_pixels": np.nan,
	        },
	        "_max_intensity_projection": {
	            "minimum_pixel_intensity": np.nan,
	            "minimum_valid_pixels_factor": np.nan,
	            "_minimum_valid_pixels": np.nan,
	        },
	        "max_intensity_projection_factors": {
	            "minimum_pixel_intensity": 4,
	            "minimum_valid_pixels_factor": 2,
	        }
	    },
	    "calculation": {
	        "minimum_pixel_intensity": 0,
	    },
	}

	validities_filename_suffix = "validities"
	mean_intensities_of_all_slices_filename_suffix = "mean_intensities_slicewise"
	mean_intensities_of_valid_slices_filename_suffix = "mean_intensities_valid_slicewise"
	mean_intensities_of_max_intensity_projection_filename_suffix = "mean_intensities_max_intensity_projection"

	def __init__(
	    self,
	    parent_image: Image,
	    validity_determining_channels: tuple[int],
	    cache_filepaths: dict,
	    thresholds: dict[str, int | float] | None = None,
	):
		"""
		Initialize the ValidityIntensityMetrics object.

		Args:
			parent_image (Image): The image for which metrics are calculated.
			validity_determining_channels (tuple[int]): Channels used to determine validity.
			thresholds (dict[str, int | float] | None): Thresholds for metrics calculations.
		"""
		self.parent_image = parent_image
		self.image_x_size = parent_image.shape[3]
		self.image_y_size = parent_image.shape[2]
		self.image_z_size = parent_image.shape[1]
		self.image_num_channels = parent_image.shape[0]

		self.validity_determining_channels = validity_determining_channels

		self._thresholds = self.default_thresholds.copy()

		if thresholds is not None:
			self.thresholds = thresholds

		self._validities: np.ndarray | None = None
		self._mean_intensities: dict[str, np.ndarray | None] = {
		    "slicewise": None,
		    "max_intensity_projection": None,
		}

		self.cache_filepaths = cache_filepaths

		self.load_from_cache()

	@property
	def thresholds(self) -> dict:
		return self._thresholds

	@thresholds.setter
	def thresholds(self, new_thresholds: dict) -> None:
		utils._recursively_update_dict(self._thresholds, new_thresholds)

		slice_min_valid_factor = self._thresholds["validity"]["slice"]["minimum_valid_pixels_factor"]
		slice_min_valid_pixels = slice_min_valid_factor * (self.image_x_size * self.image_y_size)
		self._thresholds["validity"]["slice"]["_minimum_valid_pixels"] = slice_min_valid_pixels

		mip_min_intensity_factor = self._thresholds["validity"]["max_intensity_projection_factors"][
		    "minimum_pixel_intensity"]
		mip_min_valid_factor_factor = self._thresholds["validity"]["max_intensity_projection_factors"][
		    "minimum_valid_pixels_factor"]

		mip_min_intensity = mip_min_intensity_factor * self._thresholds["validity"]["slice"]["minimum_pixel_intensity"]
		mip_min_valid_pixels_factor = mip_min_valid_factor_factor * slice_min_valid_factor
		mip_min_valid_pixels = mip_min_valid_pixels_factor * (self.image_x_size * self.image_y_size)
		self._thresholds["validity"]["_max_intensity_projection"]["minimum_pixel_intensity"] = mip_min_intensity
		self._thresholds["validity"]["_max_intensity_projection"][
		    "minimum_valid_pixels_factor"] = mip_min_valid_pixels_factor
		self._thresholds["validity"]["_max_intensity_projection"]["_minimum_valid_pixels"] = mip_min_valid_pixels

		for property_name in [
		    "mean_intensities_of_all_slices",
		    "mean_intensities_of_valid_slices",
		    "mean_intensities_of_max_intensity_projection",
		    "validities",
		]:
			try:
				del self.__dict__[property_name]
			except KeyError:
				pass

		self._compute_validities()
		self._compute_mean_intensities()

		self.cache(force_regenerate = True)

	@functools.cached_property
	def mean_intensities_of_all_slices(self) -> np.ndarray:
		if self._mean_intensities["slicewise"] is None:
			self._compute_mean_intensities()

		assert isinstance(self._mean_intensities["slicewise"], np.ndarray), "Mean intensities must be a numpy array"

		return self._mean_intensities["slicewise"]

	@functools.cached_property
	def mean_intensities_of_valid_slices(self) -> np.ndarray:
		if self._mean_intensities["slicewise"] is None:
			self._compute_mean_intensities()

		assert isinstance(self._mean_intensities["slicewise"], np.ndarray), "Mean intensities must be a numpy array"

		return self._mean_intensities["slicewise"][:, self.validities]

	@functools.cached_property
	def mean_intensities_of_max_intensity_projection(self) -> np.ndarray:
		if self._mean_intensities["max_intensity_projection"] is None:
			self._compute_mean_intensities()

		assert isinstance(self._mean_intensities["max_intensity_projection"],
		                  np.ndarray), "Mean intensities must be a numpy array"

		return self._mean_intensities["max_intensity_projection"]

	@functools.cached_property
	def validities(self) -> np.ndarray:
		if self._validities is None:
			self._compute_validities()

		assert isinstance(self._validities, np.ndarray), "Validities must be a numpy array"

		return self._validities

	def _compute_validities(self) -> None:
		"""
		Compute the validity of slices and max intensity projection for the image.

		This method calculates which slices of the image are valid based on pixel
		intensity thresholds and ensures that only the longest run of valid slices
		is retained. It also determines the validity of the maximum intensity
		projection.

		Args:
			None

		Returns:
			None
		"""

		slicewise_validities = np.full(self.image_z_size, True)
		max_intensity_projection_validity = True

		for channel_num in self.validity_determining_channels:
			pixel_intensities = self.parent_image.get_image()[channel_num, :, :, :].squeeze()
			max_intensity_projection_pixel_intensities = np.max(pixel_intensities, axis = 0)

			max_intensity_projection_minimum_pixel_intensity = self.thresholds["validity"]["_max_intensity_projection"][
			    "minimum_pixel_intensity"]
			max_intensity_projection_minimum_valid_pixels = self.thresholds["validity"]["_max_intensity_projection"][
			    "_minimum_valid_pixels"]

			filtered_max_intensity_projection_pixel_intensities = max_intensity_projection_pixel_intensities[
			    max_intensity_projection_pixel_intensities >= max_intensity_projection_minimum_pixel_intensity]

			if len(filtered_max_intensity_projection_pixel_intensities) < max_intensity_projection_minimum_valid_pixels:
				max_intensity_projection_validity = False

			if max_intensity_projection_validity == False:
				for slice_num in range(self.image_z_size):
					slicewise_validities[slice_num] = False

				continue

			for slice_num in range(self.image_z_size):
				slice_pixel_intensities = pixel_intensities[slice_num, :, :].squeeze()

				minimum_pixel_intensity = self._thresholds["validity"]["slice"]["minimum_pixel_intensity"]
				minimum_valid_pixels = self._thresholds["validity"]["slice"]["_minimum_valid_pixels"]

				filtered_pixel_intensities = slice_pixel_intensities[slice_pixel_intensities >= minimum_pixel_intensity]

				if len(filtered_pixel_intensities) < minimum_valid_pixels:
					slicewise_validities[slice_num] = False

			# Remove slices which are considered valid but are not adjacent to the longest run of valid slices (i.e. the "really valid" slices)
			is_consecutive_pair = lambda pair: pair[1] - pair[0] == 1

			all_valid_slices = [slice for slice in range(self.image_z_size) if slicewise_validities[slice] == True]
			validity_invalidity_runs = itertools.groupby(zip(all_valid_slices, all_valid_slices[1:]),
			                                             key = is_consecutive_pair)
			valid_slice_runs = [list(grouper) for validity, grouper in validity_invalidity_runs if validity is True]

			if len(valid_slice_runs) == 0:
				continue

			longest_valid_run_as_pairs = max(valid_slice_runs, key = len)
			longest_valid_run = list(set(itertools.chain.from_iterable(longest_valid_run_as_pairs)))

			for slice in range(self.image_z_size):
				if slicewise_validities[slice] == True and slice not in longest_valid_run:
					slicewise_validities[slice] = False

		self._validities = slicewise_validities

	def _compute_mean_intensities(self) -> None:
		mean_intensities = {
		    "slicewise": np.empty((self.image_num_channels, self.image_z_size)),
		    "max_intensity_projection": np.empty((self.image_num_channels)),
		}

		for channel_num in range(self.image_num_channels):
			mean_intensities["slicewise"][channel_num, :] = self._mean_intensity_of_all_slices_for_channel(
			    channel = channel_num, )

			mean_intensities["max_intensity_projection"][channel_num] = self._mean_intensity_of_xy_slice(
			    channel = channel_num,
			    max_intensity_projection = True,
			)

		self._mean_intensities = mean_intensities

	def _mean_intensity_of_all_slices_for_channel(
	    self,
	    channel: int,
	) -> np.ndarray:
		pixel_intensities = self.parent_image.get_image()[channel, :, :, :].squeeze()

		calculation_threshold = self._thresholds["calculation"]["minimum_pixel_intensity"]

		mean_intensities = np.empty(self.image_z_size)
		for slice_num in range(self.image_z_size):
			pixel_intensities_for_slice = pixel_intensities[slice_num, :, :]

			mean_intensities[slice_num] = np.nanmean(
			    pixel_intensities_for_slice[pixel_intensities_for_slice >= calculation_threshold])

		return mean_intensities

	def _mean_intensity_of_xy_slice(
	    self,
	    channel,
	    slice: int | None = None,
	    max_intensity_projection: bool = False,
	) -> float:
		if max_intensity_projection:
			pixel_intensities = self.parent_image.get_max_intensity_projection_image()[channel, :, :].squeeze()
		else:
			assert slice is not None, "Slice number must be provided for XY slice mean intensity calculation"

			pixel_intensities = self.parent_image.get_image(slice_nums = [slice])[channel, :, :, :].squeeze()

		calculation_threshold = self._thresholds["calculation"]["minimum_pixel_intensity"]

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category = RuntimeWarning)
			mean_intensity = float(np.nanmean(pixel_intensities[pixel_intensities >= calculation_threshold]))

		return mean_intensity

	def write_to_file(self, filenames_without_ext: dict, force_regenerate: bool = False) -> None:
		validities_filename = f"{filenames_without_ext['validities']}.csv"
		mean_intensities_of_all_slices_filename = f"{filenames_without_ext['mean_intensities']['all_slices']}.csv"
		mean_intensities_of_valid_slices_filename = f"{filenames_without_ext['mean_intensities']['valid_slices']}.csv"
		mean_intensities_of_max_intensity_projection_filename = f"{filenames_without_ext['mean_intensities']['max_intensity_projection']}.csv"

		if not os.path.exists(validities_filename) or force_regenerate:
			array_for_writing = np.vstack([np.arange(self.image_z_size), self.validities]).T

			headers = "slice,validity"

			np.savetxt(
			    validities_filename,
			    array_for_writing,
			    delimiter = ",",
			    header = headers,
			    comments = "",
			    fmt = "%d,%d",
			)

		if not os.path.exists(mean_intensities_of_all_slices_filename) or force_regenerate:
			array_for_writing = np.vstack([np.arange(self.image_z_size), self.mean_intensities_of_all_slices]).T

			headers = "slice," + ",".join([f"channel_{channel_num}" for channel_num in range(self.image_num_channels)])

			np.savetxt(
			    mean_intensities_of_all_slices_filename,
			    array_for_writing,
			    delimiter = ",",
			    header = headers,
			    comments = "",
			    fmt = "%d," + ",".join(["%.9e" for _ in range(self.image_num_channels)]),
			)

		if not os.path.exists(mean_intensities_of_valid_slices_filename) or force_regenerate:
			valid_slices = np.arange(self.image_z_size)[self.validities]

			array_for_writing = np.vstack([valid_slices, self.mean_intensities_of_valid_slices]).T

			headers = "slice," + ",".join([f"channel_{channel_num}" for channel_num in range(self.image_num_channels)])

			np.savetxt(
			    mean_intensities_of_valid_slices_filename,
			    array_for_writing,
			    delimiter = ",",
			    header = headers,
			    comments = "",
			    fmt = "%d," + ",".join(["%.9e" for _ in range(self.image_num_channels)]),
			)

		if not os.path.exists(mean_intensities_of_max_intensity_projection_filename) or force_regenerate:
			array_for_writing = self.mean_intensities_of_max_intensity_projection.reshape(-1, 1)

			headers = ",".join([f"channel_{channel_num}" for channel_num in range(self.image_num_channels)])

			np.savetxt(
			    mean_intensities_of_max_intensity_projection_filename,
			    array_for_writing,
			    delimiter = ",",
			    header = headers,
			    comments = "",
			    fmt = ",".join(["%.9e" for _ in range(self.image_num_channels)]),
			)

	def cache(self, force_regenerate: bool = False) -> None:
		if not force_regenerate:
			filepaths_to_check = [
			    self.cache_filepaths["validities"],
			    self.cache_filepaths["mean_intensities"]["slicewise"],
			    self.cache_filepaths["mean_intensities"]["max_intensity_projection"],
			]

			if all(os.path.exists(filepath) for filepath in filepaths_to_check):
				return

		if self._validities is None:
			self._compute_validities()

		if self._mean_intensities["slicewise"] is None or self._mean_intensities["max_intensity_projection"] is None:
			self._compute_mean_intensities()

		if os.path.exists(self.cache_filepaths["validities"]) == False or force_regenerate:
			os.makedirs(os.path.dirname(self.cache_filepaths["validities"]), exist_ok = True)

			assert isinstance(self._validities, np.ndarray)
			np.savez_compressed(
			    self.cache_filepaths["validities"],
			    self._validities,
			)

		if os.path.exists(self.cache_filepaths["mean_intensities"]["slicewise"]) == False or force_regenerate:
			os.makedirs(os.path.dirname(self.cache_filepaths["mean_intensities"]["slicewise"]), exist_ok = True)

			assert isinstance(self._mean_intensities["slicewise"], np.ndarray)
			np.savez_compressed(
			    self.cache_filepaths["mean_intensities"]["slicewise"],
			    self._mean_intensities["slicewise"],
			)

		if os.path.exists(
		    self.cache_filepaths["mean_intensities"]["max_intensity_projection"]) == False or force_regenerate:
			os.makedirs(os.path.dirname(self.cache_filepaths["mean_intensities"]["max_intensity_projection"]),
			            exist_ok = True)

			assert isinstance(self._mean_intensities["max_intensity_projection"], np.ndarray)
			np.savez_compressed(
			    self.cache_filepaths["mean_intensities"]["max_intensity_projection"],
			    self._mean_intensities["max_intensity_projection"],
			)

		if os.path.exists(self.cache_filepaths["thresholds"]) == False or force_regenerate:
			os.makedirs(os.path.dirname(self.cache_filepaths["thresholds"]), exist_ok = True)

			with open(self.cache_filepaths["thresholds"], "w") as thresholds_file:
				json.dump(self._thresholds, thresholds_file, indent = 4)

	def load_from_cache(self):
		if os.path.exists(self.cache_filepaths["validities"]):
			self._validities = np.load(self.cache_filepaths["validities"])["arr_0"]

		if os.path.exists(self.cache_filepaths["mean_intensities"]["slicewise"]):
			self._mean_intensities["slicewise"] = np.load(
			    self.cache_filepaths["mean_intensities"]["slicewise"])["arr_0"]

		if os.path.exists(self.cache_filepaths["mean_intensities"]["max_intensity_projection"]):
			self._mean_intensities["max_intensity_projection"] = np.load(
			    self.cache_filepaths["mean_intensities"]["max_intensity_projection"])["arr_0"]

		if os.path.exists(self.cache_filepaths["thresholds"]):
			with open(self.cache_filepaths["thresholds"], "r") as thresholds_file:
				self._thresholds = json.load(thresholds_file)


class ConfocalImage():
	"""
	An abstract base class for handling confocal images.

	Warnings:
		This class is abstract(ish) and should not be instantiated directly. Objects should be created using subclasses :class:`UnsectionedConfocalImage` or :class:`ConfocalImageSection`.

	Attributes:
		filename (str): The name of the image file.
		filename_without_ext (str): The filename without its extension.
		filepath (str): Relative path to the image file.
		cache_dir (str): Directory where cache files are stored.
		cache_filepaths (dict): Filepaths for cached data.
		original_image_format (str): Format of the original image file.
		inputs (Inputs | None): Sorted inputs associated with the image.
		fluorophores (list[str] | None): List of fluorophores in the image.
		hours_after_exposure (int): Hours since exposure for the image.
		bit_depth (int): Bit depth of the image.
		validity_determining_channels (tuple[int]): Channels used to determine validity.
		num_channels (int | None): Number of channels in the image.
		num_slices (int | None): Number of slices in the image.
		output_directories (dict | None): Directories for exporting metrics and slices.

	"""

	def __init__(
	        self,
	        filename: str,
	        inputs: dict[str, bool] | dict[str, tuple[int | float, str]] | Inputs | None,
	        fluorophores: list[str] | None,
	        hours_after_exposure: int,
	        original_image_format: str = "lof",
	        bit_depth: int = 2**8,
	        validity_determining_channels: tuple[int] = (2, ),
	        cache_dir: str = "./cache/",
	):
		"""
		Initialize the ConfocalImage object.

		Args:
			filename (str): The name of the image file.
			inputs (dict[str, bool] | dict[str, tuple[int | float, str]] | Inputs | None): Inputs associated with the image.
			fluorophores (list[str] | None): List of fluorophores in the image.
			hours_after_exposure (int): Hours since exposure for the image.
			original_image_format (str): Format of the original image file.
			bit_depth (int): Bit depth of the image.
			validity_determining_channels (tuple[int]): Channels used to determine validity.
			cache_dir (str): Directory for storing cached files.
		"""
		self.filename = os.path.basename(filename)
		self.filename_without_ext = os.path.splitext(self.filename)[0]
		self.filepath = os.path.relpath(os.path.join(os.path.dirname(filename), os.path.basename(filename)))

		self.cache_dir = cache_dir

		cache_subdir = os.path.join(self.cache_dir, self.filename_without_ext)

		self.cache_filepaths = {
		    "pickle": os.path.join(cache_subdir, f"{self.filename_without_ext}.pkl"),
		    "image": os.path.join(cache_subdir, f"{self.filename_without_ext}.npz"),
		    "thresholds": os.path.join(cache_subdir, f"{self.filename_without_ext}_thresholds.json"),
		    "validities": os.path.join(cache_subdir, f"{self.filename_without_ext}_validities.npz"),
		    "mean_intensities": {
		        "slicewise":
		        os.path.join(
		            cache_subdir,
		            "mean_intensities",
		            f"{self.filename_without_ext}_mean_intensities_slicewise.npz",
		        ),
		        "max_intensity_projection":
		        os.path.join(
		            cache_subdir,
		            f"{self.filename_without_ext}_mean_intensities_max_intensity_projection.npz",
		        ),
		    }
		}

		if not os.path.exists(cache_subdir):
			os.makedirs(cache_subdir, exist_ok = True)

		self.original_image_format = os.path.splitext(self.filename)[1][1:].lower()

		if inputs is not None:
			if isinstance(inputs, Inputs):
				self.inputs = inputs
			else:
				self.inputs = Inputs(inputs_dict = inputs)
		else:
			self.inputs = None

		self.fluorophores = fluorophores
		self.hours_after_exposure = hours_after_exposure

		# this cannot be inferred from the filename, as LOF files are converted and then read as NPZ files
		self.original_image_format = original_image_format.lower()
		self.bit_depth = bit_depth
		self.validity_determining_channels = validity_determining_channels
		self.num_channels = None
		self.num_slices = None

		self.output_directories = None

	@abc.abstractmethod
	def _init_image(self) -> Image:
		...

	@functools.cached_property
	def validity_intensity_metrics(self) -> ValidityIntensityMetrics:
		if not hasattr(self, "_validity_intensity_metrics"):
			self._validity_intensity_metrics = self._init_validity_intensity_metrics()

		return self._validity_intensity_metrics

	def _init_validity_intensity_metrics(self) -> ValidityIntensityMetrics:
		return ValidityIntensityMetrics(
		    parent_image = self._image,  # type: ignore[attr-defined]
		    validity_determining_channels = self.validity_determining_channels,
		    cache_filepaths = {
		        key: value
		        for key, value in self.cache_filepaths.items()
		        if key in ["validities", "mean_intensities", "thresholds"]
		    },
		)

	@abc.abstractmethod
	def _set_output_directories(self) -> None:
		...

	def write_metrics_to_file(self, force_regenerate: bool = False) -> None:
		if self.output_directories is None:
			self._set_output_directories()

		assert isinstance(self.output_directories, dict)
		filenames_dict = self.output_directories["metrics"]

		def _append_filename_to_dirs(filenames_dict: dict) -> dict:
			for key, value in filenames_dict.items():
				if isinstance(value, str):
					filenames_dict[key] = os.path.join(value, f"{self.filename_without_ext}_{key}")
				elif isinstance(value, dict):
					filenames_dict[key] = _append_filename_to_dirs(value)

			return filenames_dict

		filenames_dict = _append_filename_to_dirs(filenames_dict)

		self.validity_intensity_metrics.write_to_file(  # type: ignore[attr-defined]
		    filenames_without_ext = filenames_dict,
		    force_regenerate = force_regenerate,
		)

	def get_image(self, crop: bool | None = None, slice_num: int | None = None):
		return self._image.get_image(crop = crop, slice_nums = slice_num)  # type: ignore[attr-defined]

	@functools.cached_property
	def image(self) -> np.ndarray:
		return self._image.get_image(crop = False)  # type: ignore[attr-defined]

	@functools.cached_property
	def cropped_image(self, ) -> np.ndarray:
		return self._image.get_image(crop = True)  # type: ignore[attr-defined]

	@property  # this cannot be cached, because generators are weird and must be tee'd every time
	def sections(self) -> typing.Iterator["ConfocalImageSection"]:
		num_sections = self.num_x_sections * self.num_y_sections  # type: ignore[attr-defined]

		cache_subdir = os.path.join(self.cache_dir, self.filename_without_ext)
		filenames_to_check = [
		    os.path.join(
		        cache_subdir,
		        f"{self.filename_without_ext}_section_{section_num}_of_{num_sections}",
		        f"{self.filename_without_ext}_section_{section_num}_of_{num_sections}.pkl",
		    ) for section_num in range(num_sections)
		]

		if all(os.path.exists(filename) for filename in filenames_to_check):
			for section_pickle_filename in filenames_to_check:
				with open(section_pickle_filename, "rb") as pickle_file:
					section = pickle.load(pickle_file)

				yield section

		assert hasattr(self, "_sections"), "Sections must be initialized before accessing them"
		self._sections, sections = itertools.tee(self._sections)

		for section in sections:
			yield section

	def _dump_as_pickle(self) -> None:
		with open(self.cache_filepaths["pickle"], "wb") as pickle_file:
			pickle.dump(self, pickle_file)

	def cache(self, force_regenerate: bool = False) -> None:
		"""Cache the image and metrics to disk.

		Args:
			force_regenerate (bool, optional): If True, regenerate the cache even if it already exists. Defaults to False.
		"""
		if os.path.exists(self.cache_filepaths["pickle"]) == False or force_regenerate:
			self._dump_as_pickle()

		self.validity_intensity_metrics.cache(force_regenerate = force_regenerate)

		for property_name in [
		    "_image",
		    "_cropped_image",
		]:
			try:
				del self._image.__dict__[property_name]  # type: ignore[attr-defined]
			except KeyError:
				pass

		for property_name in ["_image", "image", "cropped_image", "_sections"]:
			try:
				del self.__dict__[property_name]
			except KeyError:
				pass

	def clear_cache(self) -> None:
		for cache_filepath in self.cache_filepaths.values():
			if os.path.exists(cache_filepath):
				os.remove(cache_filepath)

	def update_thresholds(
	    self,
	    new_thresholds: dict[str, int | float],
	    repickle: bool = True,
	):
		self.validity_intensity_metrics.thresholds = new_thresholds  # type: ignore[attr-defined]

		if repickle:
			self._dump_as_pickle()

		try:
			del self.image
		except AttributeError:
			pass

	@functools.cached_property
	def validities(self) -> np.ndarray:
		return self.validity_intensity_metrics.validities  # type: ignore[attr-defined]

	@functools.cached_property
	def mean_intensities_of_all_slices(self) -> np.ndarray:
		return self.validity_intensity_metrics.mean_intensities_of_all_slices  # type: ignore[attr-defined]

	@functools.cached_property
	def mean_intensities_of_valid_slices(self) -> np.ndarray:
		return self.validity_intensity_metrics.mean_intensities_of_valid_slices  # type: ignore[attr-defined]

	@functools.cached_property
	def mean_intensities_of_max_intensity_projection(self) -> np.ndarray:
		return self.validity_intensity_metrics.mean_intensities_of_max_intensity_projection  # type: ignore[attr-defined]

	@functools.cached_property
	def mean_intensities_across_valid_slices(self) -> np.ndarray:
		return np.mean(self.mean_intensities_of_valid_slices, axis = 1)

	def export_slices(
	    self,
	    channels: list[int] | None = None,
	    slice_nums: list[int] | None = None,
	    only_valid_slices: bool = False,
	    crop: bool = False,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	):
		assert isinstance(self.output_directories, dict), "Output directories must be set before exporting"
		assert "export" in self.output_directories.keys(), "Output directories must be set before exporting"
		assert "slicewise" in self.output_directories["export"], "Output directories must be set before exporting"

		if only_valid_slices:
			assert slice_nums is None, "Slice numbers cannot be specified when exporting only valid slices"
			slice_nums = np.arange(self.num_slices)[self.validities]  # type: ignore[attr-defined]

		self._image.export_slices(  # type: ignore[attr-defined]
		    export_base_filename = os.path.join(
		        self.output_directories["export"]["slicewise"],
		        self.filename_without_ext,
		    ),
		    export_filename_extension = "png",
		    channels = channels,
		    slice_nums = slice_nums,
		    crop = crop,
		    clear_cache = clear_cache,
		    colour_scaling_factor = colour_scaling_factor,
		)

	def export_max_intensity_projection(
	    self,
	    channels: list[int] | None = None,
	    crop: bool = False,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	):
		assert isinstance(self.output_directories, dict), "Output directories must be set before exporting"
		assert "export" in self.output_directories.keys(), "Output directories must be set before exporting"
		assert "max_intensity_projection" in self.output_directories[
		    "export"], "Output directories must be set before exporting"

		self._image.export_max_intensity_projection(  # type: ignore[attr-defined]
		    export_base_filename = os.path.join(
		        self.output_directories["export"]["max_intensity_projection"],
		        self.filename_without_ext,
		    ),
		    export_filename_extension = "png",
		    channels = channels,
		    crop = crop,
		    clear_cache = clear_cache,
		    colour_scaling_factor = colour_scaling_factor,
		)

	def export_image_stack(
	    self,
	    channels: list[int] | None = None,
	    crop: bool = False,
	    clear_cache: bool = False,
	    colour_scaling_factor: float | int = 1,
	    upsampling_factor: float | int = 1,
	    image_alpha: float | int = 1,
	    v_offset_factor: int = 4,
	    only_valid_slices: bool = False,
	    border_width: int = 2,
	):
		assert isinstance(self.output_directories, dict), "Output directories must be set before exporting"
		assert "export" in self.output_directories.keys(), "Output directories must be set before exporting"
		assert "image_stack" in self.output_directories["export"], "Output directories must be set before exporting"

		export_base_filename = os.path.join(
		    self.output_directories["export"]["image_stack"],
		    self.filename_without_ext,
		)

		slice_nums = None
		if only_valid_slices:
			slice_nums = np.arange(self.num_slices)[self.validities]  # type: ignore[attr-defined]
			export_base_filename = f"{export_base_filename}_valid_slices"

		self._image.export_image_stack(  # type: ignore[attr-defined]
		    export_base_filename = export_base_filename,
		    export_filename_extension = "png",
		    channels = channels,
		    slice_nums = slice_nums,
		    crop = crop,
		    clear_cache = clear_cache,
		    colour_scaling_factor = colour_scaling_factor,
		    upsampling_factor = upsampling_factor,
		    image_alpha = image_alpha,
		    v_offset_factor = v_offset_factor,
		    border_width = border_width,
		)


class UnsectionedConfocalImage(ConfocalImage):
	"""
	A subclass of ConfocalImage for handling unsectioned confocal images.

	Attributes:
		crop (bool): Indicates whether cropping is enabled.
		num_x_sections (int): Number of sections along the x-axis.
		num_y_sections (int): Number of sections along the y-axis.
		_sections (typing.Iterator[ConfocalImageSection]): Iterator for chopped sections.
	"""

	def __init__(
	        self,
	        filename: str,
	        inputs: dict[str, bool] | dict[str, tuple[int | float, str]] | Inputs | None,
	        fluorophores: list[str] | None,
	        hours_after_exposure: int,
	        num_x_sections: int = 8,
	        num_y_sections: int = 8,
	        original_image_format: str = "lof",
	        bit_depth: int = 2**8,
	        validity_determining_channels: tuple[int] = (2, ),
	        crop: bool = True,
	        **kwargs,
	):
		"""
		Initialize the UnsectionedConfocalImage object.

		Args:
			filename (str): Path to the image file.
			inputs (dict[str, bool] | dict[str, tuple[int | float, str]] | Inputs | None): Inputs associated with the image.
			fluorophores (list[str] | None): List of fluorophores in the image.
			hours_after_exposure (int): Hours since exposure for the image.
			num_x_sections (int): Number of sections along the x-axis.
			num_y_sections (int): Number of sections along the y-axis.
			original_image_format (str): Format of the original image file. Optional, defaults to "lof".
			bit_depth (int): Bit depth of the image. Optional, defaults to :math:`2^8`.
			validity_determining_channels (tuple[int]): Channels used to determine validity.
			crop (bool): Whether cropping is enabled.
		"""
		super().__init__(
		    filename = filename,
		    inputs = inputs,
		    fluorophores = fluorophores,
		    hours_after_exposure = hours_after_exposure,
		    original_image_format = original_image_format,
		    bit_depth = bit_depth,
		    validity_determining_channels = validity_determining_channels,
		)

		self.crop = crop

		if self.original_image_format == "czi":
			self.czi_file = czifile.CziFile(self.filepath) if self.original_image_format == "czi" else None

			# pyright cannot detect methods of lazyattrs, hence the ignore
			self.num_channels = self.czi_file.shape[1]  # type: ignore[attr-defined]
			self.num_slices = self.czi_file.shape[3]  # type: ignore[attr-defined]
			self.image_x_size = self.czi_file.shape[4]  # type: ignore[attr-defined]
			self.image_y_size = self.czi_file.shape[4]  # type: ignore[attr-defined]
		elif self.original_image_format == "lof":
			assert os.path.splitext(
			    self.filepath)[1][1:].lower() == "npz", "LOF files must be converted to NPZ files before being read"

			with zipfile.ZipFile(self.filepath) as npz_file:
				assert len(
				    npz_file.namelist()) == 1, "NPZ files converted from LOF files must contain exactly one object"

				with npz_file.open(npz_file.namelist()[0]) as npy_file:
					version = np.lib.format.read_magic(npy_file)
					shape, _, _ = np.lib.format._read_array_header(npy_file, version)  # type: ignore[attr-defined]

					self.num_channels = shape[0]
					self.num_slices = shape[1]
					self.image_x_size = shape[2]
					self.image_y_size = shape[3]
		elif self.original_image_format == "npz":
			with np.load(self.filepath) as npz_file:
				shape = npz_file["arr_0"].shape
				self.num_channels = shape[0]
				self.num_slices = shape[1]
				self.image_x_size = shape[2]
				self.image_y_size = shape[3]
		else:
			raise ValueError(f"Unsupported image format: {self.original_image_format}")

		self._set_output_directories()

		self.num_x_sections = num_x_sections
		self.num_y_sections = num_y_sections

		self._sections = self.chop_up()

	def __reduce__(self):
		return (
		    self.__class__,
		    (
		        self.filepath,  # yes, this must be self.filepath, not self.filename
		        self.inputs,
		        self.fluorophores,
		        self.hours_after_exposure,
		        self.num_x_sections,
		        self.num_y_sections,
		        self.original_image_format,
		        self.bit_depth,
		        self.validity_determining_channels,
		        self.crop,
		    ),
		)

	@property
	def _image(self) -> Image:
		return self._init_image(filepath = self.filepath, crop = self.crop, bit_depth = self.bit_depth)

	def _init_image(  # type: ignore[override]
	        self,
	        filepath: str,
	        crop: bool = False,
	        bit_depth: int = 2**8,
	) -> Image:
		"""
		Initialize the image object.

		Args:
			filepath (str): Path to the image file.
			crop (bool): Whether to crop the image.
			bit_depth (int): Bit depth of the image.

		Returns:
			An :any:`Image` object initialized with the given parameters.
		"""
		return Image(filename = filepath, crop = crop, bit_depth = bit_depth)

	def _set_output_directories(self) -> None:
		"""
		Set directory paths for exporting images and metrics, and create them if they do not exist.
		"""
		self.output_directories = {
		    "export": {
		        "slicewise": "./exports/slicewise/unsectioned_images/",
		        "max_intensity_projection": "./exports/maximum_intensity_projections/unsectioned_images/",
		        "image_stack": "./exports/image_stacks/unsectioned_images/",
		    },
		    "metrics": {
		        "validities": "./metrics/validities/unsectioned_images/",
		        "mean_intensities": {
		            "all_slices": "./metrics/mean_intensities/unsectioned_images/all_slices/",
		            "valid_slices": "./metrics/mean_intensities/unsectioned_images/valid_slices/",
		            "max_intensity_projection":
		            "./metrics/mean_intensities/unsectioned_images/max_intensity_projection/",
		        },
		    },
		}

		utils.recursively_create_directories(self.output_directories)

	def chop_up(self) -> typing.Iterator["ConfocalImageSection"]:
		""" Chop up the image in the :math:`x` and :math:`y` directions.

		Chop up the image into sections along the :math:`x` and :math:`y` axes, yielding sections as ConfocalImageSection objects.
		The number of sections is determined by :any:`num_x_sections` and :any:`num_y_sections`.

		Yields:
			ConfocalImageSection: A section of the image, represented as a ConfocalImageSection object.

		"""
		if self.crop:
			full_image = self.cropped_image
			del self.cropped_image
		else:
			full_image = self.image
			del self.image

		image_x_sections = np.array_split(full_image, self.num_x_sections, axis = 2)
		image_sections = list(
		    itertools.chain(
		        *[np.array_split(x_section, self.num_y_sections, axis = 3) for x_section in image_x_sections]))

		del image_x_sections
		del full_image

		for section_num, section_image in enumerate(image_sections):
			yield ConfocalImageSection(
			    image = section_image,
			    source_image = self,
			    section_num = section_num,
			    total_sections = self.num_x_sections * self.num_y_sections,
			)


class ConfocalImageSection(ConfocalImage):
	"""
	A subclass of ConfocalImage representing sections of confocal images.

	Attributes:
		source_image (UnsectionedConfocalImage): The unsectioned image from which this section originated.
		section_num (int): The section number.
		total_sections (int): The total number of sections.
	"""

	def __init__(
	    self,
	    image: np.ndarray | None,  # the None is required for reducing with __reduce__
	    source_image: UnsectionedConfocalImage,
	    section_num: int,
	    total_sections: int,
	):
		"""
		Initialize a ConfocalImageSection object.

		Args:
			image (np.ndarray | None): Numpy array representing the image section.
			source_image (UnsectionedConfocalImage): Source unsectioned image.
			section_num (int): The number of this section.
			total_sections (int): Total number of sections.
		"""
		self.source_image = source_image
		self.section_num = section_num
		self.total_sections = total_sections

		super().__init__(
		    filename = f"{source_image.filename_without_ext}_section_{section_num}_of_{total_sections}.npz",
		    inputs = source_image.inputs,
		    fluorophores = source_image.fluorophores,
		    hours_after_exposure = source_image.hours_after_exposure,
		    original_image_format = source_image.original_image_format,
		    bit_depth = source_image.bit_depth,
		    validity_determining_channels = source_image.validity_determining_channels,
		    cache_dir = os.path.join(source_image.cache_dir, source_image.filename_without_ext),
		)

		self._image = self._init_image(image = image)

		self._set_output_directories()

		self.num_channels = self.source_image.num_channels
		self.num_slices = self.source_image.num_slices

	def __reduce__(self):
		return (
		    self.__class__,
		    (
		        None,
		        self.source_image,
		        self.section_num,
		        self.total_sections,
		    ),
		)

	def _init_image(self, image: np.ndarray | None = None) -> Image:
		"""
		Initialize the image object.

		Args:
			image (np.ndarray | None): Numpy array representing the image section. If None, the image will be loaded from cache. If not None, the image will be saved to cache.

		Returns:
			An :any:`Image` object initialized with the given parameters.
		"""
		if image is not None and os.path.exists(self.cache_filepaths["image"]) == False:
			np.savez_compressed(self.cache_filepaths["image"], image)

		if image is not None:
			shape = np.array(image.shape).astype(int)
		else:
			shape = np.array([
			    self.source_image.num_channels,
			    self.source_image.num_slices,
			    self.source_image.image_x_size / self.source_image.num_x_sections,
			    self.source_image.image_y_size / self.source_image.num_y_sections,
			]).astype(int)

		return Image(
		    filename = self.cache_filepaths["image"],
		    crop = False,  # sections are never cropped
		    shape = shape,
		)

	def _set_output_directories(self) -> None:
		self.output_directories = {
		    "export": {
		        "slicewise": "./exports/slicewise/sections/",
		        "max_intensity_projection": "./exports/maximum_intensity_projections/sections/",
		        "image_stack": "./exports/image_stacks/sections/",
		    },
		    "metrics": {
		        "validities": "./metrics/validities/sections/",
		        "mean_intensities": {
		            "all_slices": "./metrics/mean_intensities/sections/all_slices/",
		            "valid_slices": "./metrics/mean_intensities/sections/valid_slices/",
		            "max_intensity_projection": "./metrics/mean_intensities/sections/max_intensity_projection/",
		        },
		    },
		}

		utils.recursively_create_directories(self.output_directories)
