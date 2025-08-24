#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import concurrent.futures
import functools
import itertools
import multiprocessing
import operator
import os
import time
import typing

import tqdm  # type: ignore[import-untyped]

from . import classes, utils


def _cache_unsectioned_images(
    images: typing.Iterable[classes.UnsectionedConfocalImage],
    force_regenerate: bool = False,
    use_concurrency: bool = True,
    max_workers: int = 12,
):
	images, _images = itertools.tee(images)
	num_images = sum(1 for _ in _images)
	del _images

	with tqdm.tqdm(
	    total = num_images,
	    desc = "Caching unsectioned images",
	    unit = "image",
	    leave = True,
	) as pbar:
		if use_concurrency:
			with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
				futures = {
				    executor.submit(classes.UnsectionedConfocalImage.cache, image, force_regenerate)
				    for image in images
				}

				for future in concurrent.futures.as_completed(futures):
					if future.exception() is not None:
						tqdm.tqdm.write(f"Error caching image: {future.exception()}")
					pbar.update(1)
		else:
			for image in images:
				image.cache()
				pbar.update(1)


def _cache_sections_of_image(
    image: classes.UnsectionedConfocalImage,
    force_regenerate: bool = False,
    use_concurrency: bool = True,
    max_workers: int = 8,
):
	start_time = time.time()
	with tqdm.tqdm(
	    total = image.num_x_sections * image.num_y_sections,
	    desc = f"Caching sections of {image.filename}",
	    unit = "section",
	    leave = False,
	    position = multiprocessing.current_process()._identity[0],  # type: ignore
	) as pbar:
		if use_concurrency:
			with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
				futures = executor.map(
				    operator.methodcaller("cache", force_regenerate),
				    image.sections,
				)

				for future in futures:
					try:
						print(future.result())
					except Exception:
						import traceback
						tqdm.tqdm.write(f"Error caching section: {traceback.format_exc()}")
					pbar.update(1)

		else:
			for section in image.sections:
				try:
					section.cache(force_regenerate)
				except Exception as e:
					tqdm.tqdm.write(f"Error caching section {section.filename}: {e}")
					import traceback
					tqdm.tqdm.write(traceback.format_exc())
				finally:
					pbar.update(1)

		del image


def cache_images(
    images: typing.Iterable[classes.UnsectionedConfocalImage],
    num_images: int | None = None,
    force_regenerate: bool = False,
    use_concurrency: bool = True,
    max_workers_unsectioned: int = 12,
    max_workers_sections: int = 8,
):

	images, _images = itertools.tee(images)

	with tqdm.tqdm(
	    total = num_images,
	    desc = "Caching sections of unsectioned images",
	    unit = "image",
	    leave = True,
	    position = tqdm.tqdm._get_free_pos(),  # type: ignore
	) as pbar:
		with concurrent.futures.ProcessPoolExecutor(
		    max_workers = 8,
		    initializer = utils._process_terminator,
		    initargs = (os.getpid(), ),
		) as executor:
			_cache_sections_of_image_partial = functools.partial(
			    _cache_sections_of_image,
			    force_regenerate = force_regenerate,
			    use_concurrency = False,
			    max_workers = max_workers_sections,
			)
			futures = executor.map(
			    _cache_sections_of_image_partial,
			    _images,
			    chunksize = 8,
			)

			for future in futures:
				pbar.update(1)

	images, _images = itertools.tee(images)

	_cache_unsectioned_images(
	    images = images,
	    force_regenerate = force_regenerate,
	    use_concurrency = use_concurrency,
	    max_workers = max_workers_unsectioned,
	)
