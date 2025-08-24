#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import concurrent.futures
import os
import time
import typing

import matplotlib.patches as mpatches  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy  # type: ignore[import-untyped]
import tol_colors as tc  # type: ignore[import-untyped]
import tqdm  # type: ignore[import-untyped]

from . import classes, utils

COLOUR_SET = tc.tol_cset("bright")
LINEWIDTH = 1
HINT_LINEWIDTH = 0.5


class MeanIntensities:

	def __init__(
	    self,
	    sections: list[classes.ConfocalImage],
	    variable_class_for_series: typing.Type[classes.Inputs] | typing.Type[classes.Fluorophores],
	    variable_for_x_axis: typing.Literal["time", "concentration"],
	    use_maximum_intensity_projection: bool = False,
	):
		assert len(sections) > 0, "At least one section is required"

		self.sections = sorted(
		    sections,
		    key = lambda section: (section.hours_after_exposure, section.inputs, section.fluorophores),
		)
		self.use_maximum_intensity_projection = use_maximum_intensity_projection

		assert sections[0].num_channels is not None, "Sections must have num_channels defined"
		self.num_channels: int = sections[0].num_channels

		self.inputs = None
		self.input_combinations = None
		self.fluorophores = None
		self.timepoints = None
		self.concentrations = None
		self.series_variables = None

		self.variable_class_for_series = variable_class_for_series
		if self.variable_class_for_series == classes.Inputs:
			self.input_combinations = utils.unique([section.inputs for section in self.sections])
			self.series_variables = self.input_combinations
		elif self.variable_class_for_series == classes.Fluorophores:
			self.fluorophores = utils.unique([section.fluorophores for section in self.sections])
			self.series_variables = self.fluorophores
		else:
			raise ValueError("variable_class_for_series must be Inputs or Fluorophores")

		assert self.series_variables is not None, "Series variables must be set"

		self.variable_for_x_axis = variable_for_x_axis
		if self.variable_for_x_axis == "time":
			self.timepoints = utils.unique([section.hours_after_exposure for section in self.sections])
		elif self.variable_for_x_axis == "concentration":
			self.inputs = utils.unique(
			    [section.inputs.input_names for section in self.sections if section.inputs is not None])

			self.concentrations = utils.unique(
			    [section.inputs.input_values for section in self.sections if section.inputs is not None])

		self.x_values: np.ndarray | None = None
		self.y_values: np.ndarray | None = None
		self.y_values_bootstrap_confidence_intervals: np.ndarray | None = None
		self.num_valid_samples: np.ndarray | None = None

		if self.variable_class_for_series == classes.Inputs:
			self.get_series_variable_value = lambda section: section.inputs
		elif self.variable_class_for_series == classes.Fluorophores:
			self.get_series_variable_value = lambda section: section.fluorophores
		else:
			raise ValueError("variable_class_for_series must be Inputs or Fluorophores")

		if self.variable_for_x_axis == "time":
			self.get_x_value = lambda section: section.hours_after_exposure
		elif self.variable_for_x_axis == "concentration":
			self.get_x_value = lambda section: section.inputs.input_values[0] if section.inputs is not None else None
		else:
			raise ValueError("variable_for_x_axis must be 'time' or 'concentration'")

	def plot_linegraphs(
	    self,
	    channels: list[int] | None = None,
	    normalize_across_channels: bool = True,
	    normalization_channel: int | None = None,
	    normalize_across_series: bool = False,
	    normalization_series_variable_value: classes.Inputs | classes.Fluorophores | None = None,
	    output_filename: str = "linegraphs",
	    figsize: tuple[float, float] = (0.8 * 8.27, 0.3 * 11.69),
	    x_tick_frequency: int = 5,
	    legend_loc: str = "outside right upper",
	    nth_derivative: int = 0,
	    mark_confidence_intervals: bool = True,
	    xlabel: str | None = None,
	    ylabels: str | list[str] | None = None,
	) -> None:
		assert self.series_variables is not None

		if normalize_across_channels and normalization_channel is None:
			normalization_channel = self.sections[0].validity_determining_channels[0]

		self.load_from_cache()

		if channels is None:
			assert self.sections[0].num_channels is not None, "No channels specified and no channels in section"
			channels = list(range(self.sections[0].num_channels))

		if normalize_across_channels and normalization_channel is None:
			normalization_channel = self.sections[0].validity_determining_channels[0]

		if normalize_across_channels:
			output_filename = f"{output_filename}_normalized_to_channel_{normalization_channel}"

		if self.x_values is None or self.y_values is None or self.y_values_bootstrap_confidence_intervals is None or self.num_valid_samples is None:
			channels_to_populate = channels.copy()
			if normalization_channel is not None:
				channels_to_populate.append(normalization_channel)

			self.populate_values(
			    channels = channels_to_populate,
			    normalize_across_channels = normalize_across_channels,
			    normalization_channel = normalization_channel,
			)

		if normalize_across_series and normalization_series_variable_value is None:
			raise ValueError("normalization_series_variable_value must be set if normalize_across_series is True")

		if normalize_across_series:
			assert self.series_variables is not None, "Series variables must be set"
			normalization_series_variable_value_idx = self.series_variables.index(normalization_series_variable_value)

			if self.variable_class_for_series == classes.Inputs:
				output_filename = f"{output_filename}_normalized_to_input_combination_{normalization_series_variable_value_idx}"
			elif self.variable_class_for_series == classes.Fluorophores:
				output_filename = f"{output_filename}_normalized_to_fluorophore_{normalization_series_variable_value_idx}"

		fig = plt.figure(
		    figsize = figsize,
		    dpi = 300,
		    layout = "constrained",
		)

		ax = fig.subplots(
		    len(channels),
		    1,
		    sharex = True,
		    sharey = False,
		)

		if xlabel is None:
			match self.variable_for_x_axis:
				case "time":
					xlabel = "Hours after exposure"
				case "concentration":
					xlabel = "Concentration (units)"
				case _:
					raise ValueError("variable_for_x_axis must be 'time' or 'concentration'")

		independent_ylabels = False
		ylabel = None
		if ylabels is None:
			match nth_derivative:
				case 0:
					ylabel = "Median fluorescence intensity"
				case 1:
					ylabel = "Derivative of median fluorescence intensity"
				case 2:
					ylabel = "Second derivative of median fluorescence intensity"
				case 3:
					ylabel = "Third derivative of median fluorescence intensity"
				case _:
					ylabel = f"{nth_derivative}th derivative of median fluorescence intensity"

			if normalize_across_channels:
				ylabel += " (notmalized to reference fluorophore)"
		else:
			if isinstance(ylabels, str):
				ylabel = ylabels
			elif isinstance(ylabels, list) and len(ylabels) == len(channels):
				independent_ylabels = True
			else:
				raise ValueError("ylabels must be a string or a list of strings with the same length as channels")

		for channel_idx, channel in enumerate(channels):
			if len(channels) == 1:
				current_axes = ax
			else:
				current_axes = ax[channel_idx]

			assert isinstance(current_axes, plt.Axes)

			for series_variable_idx, series_variable_value in enumerate(self.series_variables):
				colour = COLOUR_SET[series_variable_idx % len(COLOUR_SET)]

				assert self.x_values is not None
				assert self.y_values is not None
				assert self.y_values_bootstrap_confidence_intervals is not None

				y_values = self.y_values[:, series_variable_idx, channel].copy()
				y_values_bootstrap_confidence_intervals = self.y_values_bootstrap_confidence_intervals[
				    :,
				    series_variable_idx,
				    channel,
				    :,
				].copy()

				if normalize_across_series:
					normalization_series_variable_value_idx = self.series_variables.index(
					    normalization_series_variable_value)

					normalization_y_values = self.y_values[:, normalization_series_variable_value_idx, channel].copy()
					normalization_y_values = scipy.interpolate.CubicSpline(
					    self.x_values[np.where(~np.isnan(normalization_y_values))],
					    normalization_y_values[np.where(~np.isnan(normalization_y_values))],
					)(self.x_values)

					print(f"{y_values = }")
					print(f"{normalization_y_values = }")

					y_values /= normalization_y_values

					normalization_y_values_bootstrap_confidence_intervals = self.y_values_bootstrap_confidence_intervals[
					    :,
					    normalization_series_variable_value_idx,
					    channel,
					    :,
					].copy()

					for bound_num in range(2):
						spline = scipy.interpolate.CubicSpline(
						    self.x_values[np.where(
						        ~np.isnan(normalization_y_values_bootstrap_confidence_intervals[:, bound_num])
						        & np.isfinite(normalization_y_values_bootstrap_confidence_intervals[:, bound_num]))],
						    normalization_y_values_bootstrap_confidence_intervals[np.where(
						        ~np.isnan(normalization_y_values_bootstrap_confidence_intervals[:, bound_num])
						        & np.isfinite(normalization_y_values_bootstrap_confidence_intervals[:, bound_num])),
						                                                          bound_num].squeeze(),
						)

						print(f"{spline(self.x_values).shape = }")

						y_values_bootstrap_confidence_intervals[:, bound_num] = spline(self.x_values)

					# yes, the high-low mismatch is intentional, to compute worse case confidence intervals
					y_values_bootstrap_confidence_intervals[:,
					                                        0] /= normalization_y_values_bootstrap_confidence_intervals[:,
					                                                                                                    1]
					y_values_bootstrap_confidence_intervals[:,
					                                        1] /= normalization_y_values_bootstrap_confidence_intervals[:,
					                                                                                                    0]

				x_values_for_plot = self.x_values[np.where(~np.isnan(y_values))]
				y_values_for_plot = y_values[np.where(~np.isnan(y_values))]
				y_values_bootstrap_confidence_intervals_for_plot = y_values_bootstrap_confidence_intervals[
				    np.where(~np.isnan(y_values)), :].squeeze()

				if nth_derivative > 0:
					assert len(x_values_for_plot) > 0, "No valid x values for nth derivative"
					assert len(y_values_for_plot) > 0, "No valid y values for nth derivative"

					for _ in range(nth_derivative):
						if len(x_values_for_plot) < 2:
							break

						y_values_for_plot = np.gradient(y_values_for_plot)

				current_axes.plot(
				    x_values_for_plot,
				    y_values_for_plot,
				    color = colour,
				    linestyle = "-",
				    linewidth = LINEWIDTH,
				)

				if mark_confidence_intervals:
					if len(y_values_for_plot) > 1:
						current_axes.fill_between(
						    x_values_for_plot,
						    y_values_bootstrap_confidence_intervals_for_plot[:, 0].squeeze(),
						    y_values_bootstrap_confidence_intervals_for_plot[:, 1].squeeze(),
						    edgecolor = "none",
						    facecolor = colour,
						    alpha = 0.2,
						)

			assert self.x_values is not None

			current_axes.set_xticks(
			    np.round(
			        np.arange(
			            0,
			            (np.ceil(max(self.x_values) / x_tick_frequency) + 1) * x_tick_frequency,
			            x_tick_frequency,
			        )))

			current_axes.grid(visible = True, axis = "both", which = "major")

			if independent_ylabels:
				assert isinstance(ylabels, list) and len(ylabels) == len(channels)
				current_axes.set_ylabel(ylabels[channel_idx])

		fig.supxlabel(xlabel)

		if not independent_ylabels:
			assert isinstance(ylabel, str)
			fig.supylabel(ylabel)

		legend_entries = self._get_labels()
		fig.legend(*zip(*legend_entries), loc = legend_loc)

		fig.savefig(
		    fname = f"{output_filename}.png",
		    dpi = 300,
		    format = "png",
		    bbox_inches = "tight",
		)

		fig.savefig(
		    fname = f"{output_filename}.pdf",
		    dpi = 300,
		    format = "pdf",
		    bbox_inches = "tight",
		)

		plt.close(fig)

	def _get_labels(self) -> list[tuple[mpatches.Patch, str]]:
		assert self.series_variables is not None

		labels = []

		for series_variable_value, colour in zip(self.series_variables, COLOUR_SET):
			patch = mpatches.Patch(color = colour)
			labels.append((patch, series_variable_value.pretty_str(format = "legend")))

		return labels

	def clear_values(self):
		self.x_values = None
		self.y_values = None
		self.y_values_bootstrap_confidence_intervals = None

	def initialize_values(self, ):
		x_values = None

		if self.variable_class_for_series == classes.Inputs:
			assert self.input_combinations is not None, "Input combinations must be set for Inputs variable class"
			num_series = len(self.input_combinations)
		elif self.variable_class_for_series == classes.Fluorophores:
			assert self.fluorophores is not None, "Fluorophores must be set for Fluorophores variable class"
			num_series = len(self.fluorophores)
		else:
			raise ValueError("variable_class must be Inputs or Fluorophores")

		if self.variable_for_x_axis == "time":
			x_values = np.array(self.timepoints)
		elif self.variable_for_x_axis == "concentration":
			assert self.inputs is not None, "Inputs must be set for concentration variable class"
			assert self.concentrations is not None, "Concentrations must be set for concentration variable class"

			if len(self.inputs) == 1:
				x_values = np.array(self.concentrations[0])  # TODO: Handle standardization of inputs (across units)
			else:
				NotImplementedError("Concentration with multiple inputs is not implemented yet")
		else:
			raise ValueError("variable_for_x_axis must be 'time' or 'concentration'")

		assert isinstance(x_values, np.ndarray), "x_values must be a numpy array"
		num_x_values = x_values.shape[0]

		self.x_values = x_values
		self.y_values = np.full((num_x_values, num_series, self.num_channels), np.nan)
		self.y_values_bootstrap_confidence_intervals = np.full((num_x_values, num_series, self.num_channels, 2), np.nan)
		self.num_valid_samples = np.zeros((num_x_values, num_series, self.num_channels), dtype = int)

	def populate_values(
	    self,
	    channels: list[int],
	    normalize_across_channels: bool = True,
	    normalization_channel: int | None = None,
	):
		if self.x_values is None or self.y_values is None or self.y_values_bootstrap_confidence_intervals is None or self.num_valid_samples is None:
			self.initialize_values()

		if normalize_across_channels and normalization_channel is None:
			raise ValueError("normalization_channel must be set if normalize_across_channels is True")

		assert self.series_variables is not None, "Series variables must be set"
		assert self.x_values is not None, "x_values must be set"
		assert self.y_values is not None, "y_values must be set"
		assert self.y_values_bootstrap_confidence_intervals is not None, "y_values_bootstrap_confidence_intervals must be set"
		assert self.num_valid_samples is not None, "num_valid_samples must be set"

		self.grouped_sections = self._group_sections()

		with tqdm.tqdm(
		    total = len(self.series_variables) * len(self.x_values) * self.num_channels,
		    desc = "Computing metrics for linegraph",
		    unit = "section",
		    leave = False,
		) as pbar:
			executor_jobs = []
			for series_variable_idx, series_variable_value in enumerate(self.series_variables):
				for channel_idx, channel in enumerate(range(self.num_channels)):
					if channel not in channels:
						continue

					for x_value_idx, x_value in enumerate(self.x_values):
						executor_jobs.append((
						    series_variable_idx,
						    series_variable_value,
						    channel_idx,
						    channel,
						    x_value_idx,
						    x_value,
						    normalize_across_channels,
						    normalization_channel,
						))
			with concurrent.futures.ThreadPoolExecutor() as executor:
				futures = {executor.submit(
				    self._compute_metrics_for_linegraph,
				    *job,
				)
				           for job in executor_jobs}
				for future in concurrent.futures.as_completed(futures):
					if future.exception() is not None:
						tqdm.tqdm.write(f"Error computing metrics: {future.exception()}")

					result = future.result()

					if result is not None:
						series_variable_idx = result["series_variable_idx"]
						series_variable_value = result["series_variable_value"]
						channel_idx = result["channel_idx"]
						channel = result["channel"]
						x_value_idx = result["x_value_idx"]
						x_value = result["x_value"]
						y_value = result["y_value"]
						y_value_bootstrap_confidence_interval = result["y_value_bootstrap_confidence_interval"]
						num_valid_samples = result["num_valid_samples"]

						self.y_values[x_value_idx, series_variable_idx, channel_idx] = y_value
						self.y_values_bootstrap_confidence_intervals[
						    x_value_idx,
						    series_variable_idx,
						    channel_idx,
						    :,
						] = y_value_bootstrap_confidence_interval

						self.num_valid_samples[x_value_idx, series_variable_idx, channel_idx] = num_valid_samples

					pbar.update(1)

		self.cache()

	def cache(self):
		cache_filenames = {
		    "x_values": "./cache/mean_intensities/x_values.npz",
		    "y_values": "./cache/mean_intensities/y_values.npz",
		    "y_values_bootstrap_confidence_intervals":
		    "./cache/mean_intensities/y_values_bootstrap_confidence_intervals.npz",
		    "num_valid_samples": "./cache/mean_intensities/num_valid_samples.npz",
		}

		if not os.path.exists("./cache/mean_intensities"):
			os.makedirs("./cache/mean_intensities", exist_ok = True)

		if self.x_values is not None:
			np.savez_compressed(
			    cache_filenames["x_values"],
			    self.x_values,
			)

		if self.y_values is not None:
			np.savez_compressed(
			    cache_filenames["y_values"],
			    self.y_values,
			)

		if self.y_values_bootstrap_confidence_intervals is not None:
			np.savez_compressed(
			    cache_filenames["y_values_bootstrap_confidence_intervals"],
			    self.y_values_bootstrap_confidence_intervals,
			)

		if self.num_valid_samples is not None:
			np.savez_compressed(
			    cache_filenames["num_valid_samples"],
			    self.num_valid_samples,
			)

	def load_from_cache(self):
		cache_filenames = {
		    "x_values": "./cache/mean_intensities/x_values.npz",
		    "y_values": "./cache/mean_intensities/y_values.npz",
		    "y_values_bootstrap_confidence_intervals":
		    "./cache/mean_intensities/y_values_bootstrap_confidence_intervals.npz",
		    "num_valid_samples": "./cache/mean_intensities/num_valid_samples.npz",
		}

		if os.path.exists(cache_filenames["x_values"]):
			self.x_values = np.load(cache_filenames["x_values"])["arr_0"]

		if os.path.exists(cache_filenames["y_values"]):
			self.y_values = np.load(cache_filenames["y_values"])["arr_0"]

		if os.path.exists(cache_filenames["y_values_bootstrap_confidence_intervals"]):
			self.y_values_bootstrap_confidence_intervals = np.load(
			    cache_filenames["y_values_bootstrap_confidence_intervals"])["arr_0"]

		if os.path.exists(cache_filenames["num_valid_samples"]):
			self.num_valid_samples = np.load(cache_filenames["num_valid_samples"])["arr_0"]

	def _group_sections(self):
		grouped_sections = {}
		for section in self.sections:
			series_variable_value = self.get_series_variable_value(section)
			x_value = self.get_x_value(section)

			grouping_key = (series_variable_value, x_value)

			if grouping_key not in grouped_sections:
				grouped_sections[grouping_key] = []
			grouped_sections[grouping_key].append(section)

		return grouped_sections

	def _compute_metrics_for_linegraph(
	        self,
	        series_variable_idx: int,
	        series_variable_value: classes.Inputs | classes.Fluorophores,
	        channel_idx: int,
	        channel: int,
	        x_value_idx: int,
	        x_value: float | int,
	        normalize_across_channels: bool,
	        normalization_channel: int | None,
	        confidence_level: float = 0.95,
	        bootstrap_repetitions: int = int(1e5),
	) -> dict:
		valid_mean_intensities = []

		sections = self.grouped_sections.get((series_variable_value, x_value), [])

		for section in sections:
			if self.use_maximum_intensity_projection:
				mean_intensities = section.mean_intensities_of_max_intensity_projection
			else:
				mean_intensities = section.mean_intensities_across_valid_slices

			mean_intensity = mean_intensities[channel_idx]

			if normalize_across_channels and normalization_channel is not None:
				normalization_mean_intensity = mean_intensities[normalization_channel]

				if normalization_mean_intensity != 0:
					mean_intensity /= normalization_mean_intensity
				else:
					mean_intensity = np.nan

			valid_mean_intensities.append(mean_intensity)

		valid_mean_intensities = [intensity for intensity in valid_mean_intensities if not np.isnan(intensity)]

		y_value = np.nan
		y_value_bootstrap_confidence_interval = (np.nan, np.nan)

		if len(valid_mean_intensities) > 0:
			y_value = np.median(valid_mean_intensities)

		valid_mean_intensities = np.array(valid_mean_intensities)

		if len(valid_mean_intensities) > 1:
			y_value_bootstrap_result = scipy.stats.bootstrap(
			    (valid_mean_intensities, ),
			    np.median,
			    confidence_level = confidence_level,
			    n_resamples = bootstrap_repetitions,
			)

			y_value_bootstrap_confidence_interval = (
			    y_value_bootstrap_result.confidence_interval.low,
			    y_value_bootstrap_result.confidence_interval.high,
			)

		return {
		    "series_variable_idx": series_variable_idx,
		    "series_variable_value": series_variable_value,
		    "channel_idx": channel_idx,
		    "channel": channel,
		    "x_value_idx": x_value_idx,
		    "x_value": x_value,
		    "y_value": y_value,
		    "y_value_bootstrap_confidence_interval": y_value_bootstrap_confidence_interval,
		    "num_valid_samples": len(valid_mean_intensities),
		}


class MeanIntensitiesOverTime(MeanIntensities):

	def __init__(
	    self,
	    sections: list[classes.ConfocalImage],
	    use_maximum_intensity_projection: bool = False,
	):
		super().__init__(
		    sections = sections,
		    variable_class_for_series = classes.Inputs,
		    variable_for_x_axis = "time",
		    use_maximum_intensity_projection = use_maximum_intensity_projection,
		)


class MeanIntensitiesAcrossConcentration(MeanIntensities):

	def __init__(
	    self,
	    sections: list[classes.ConfocalImage],
	    use_maximum_intensity_projection: bool = False,
	):
		super().__init__(
		    sections = sections,
		    variable_class_for_series = classes.Inputs,
		    variable_for_x_axis = "concentration",
		    use_maximum_intensity_projection = use_maximum_intensity_projection,
		)
