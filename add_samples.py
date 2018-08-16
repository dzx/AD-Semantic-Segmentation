#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:02:42 2018

@author: dzx
"""
import os.path
from os import scandir, DirEntry
from pathlib import Path
import math
def link_sample(source, dest, class_width, samp_dir, mask_dir, file_name, 
                prefix='', id_offset=0):
    """Creates symbolic links to sample/mask image pairs in source directory constructing link names based on 
    original names, given prefix and numerical offset. Both source and destination directories should have
    sample and mask subdirectories. Source image and mask files should be placed in ther respective subdirectiries 
    with same file name. Symbolic link name will be constructed as prefix+sample_id+id_offset.
    sample_id is a number constructed by gathering all digits from sample file name,
    id_offset is added if non-zero, leading zeros are prepended to match class_width, 
    and prefix is prepended if non-empty
    Parameters:
        source : source directory
        dest : destination directory
        class_width: total number of digits in link name
        samp_dir : sample directory name
        mask_dir : mask directory name
        file_name : name of sample image/mask file
        prefix : to prepend to link name
        id_offset : to add to derived sample number 
    """
    sample_id, ext = os.path.splitext(file_name)
    sample_id = ''.join([c for c in sample_id if c.isdigit()])
    sample_num = int(sample_id) + id_offset
    new_name = prefix
    new_name += format(sample_num, '0'+str(class_width)+'d')
    new_name += ext
    for dest_dir in [samp_dir, mask_dir]:
        src = Path(source, dest_dir, file_name)
        dst = Path(dest, dest_dir, new_name)
        dst.symlink_to(src.resolve())

def link_samples(source, dest, class_widths, class_counts, samp_dir, mask_dir, 
                 prefix='', id_offset=0):
    """Creates symbolic links to image-mask pairs from source directory in destination directory 
    traversing subdirectories. Subdirectores under source are considered to represent class hierarchy.
    Subdirectories at the bottom of class hierarchy should contain separate sample and mask subdirectory
    with identically named files for image-mask pair in each of them. Syminks will be created in 2 
    correspondig subdirectories of destination directory, flattening the original directory structure
    while preserving information about class membership as follows. Final link name will consist
    of concatenated class-subclass... numbers ending with absolute number of sample in its class.
    Absolute sample number consist of digits from original file name, incremented by total number of
    samples of same class processed in earier directories, and offset if any. If directory contains
    more subdirectories than expected number of classes on that level, addictional directories will
    be assumed class number as directory_ordinal_number % class_count. (sub)class/sample numbers
    will be zero-padded to match specified width
    Parameters:
        source : source directory
        dest : destination directory
        class_widths : withs for class numbers per hierarchy level
        class_counts : numbers of classes for each hierarchy level
        samp_dir : sample subdirectory name
        mask_dir : mask subdirectory name
        prefix : link prefix
        id_offset : number by which to in/de-crement original sample number
    """
    dirs = []
    found_samp, found_mask = False, False
    result = 0
    for entry in os.scandir(source):
        if entry.is_dir():
            if entry.name == samp_dir:
                found_samp = True
            elif entry.name == mask_dir:
                found_mask = True
            else:
                dirs.append(entry.name)
    if found_samp and found_mask:
        samples = os.path.join(source, samp_dir)
        for sample in os.listdir(samples):
            link_sample(source, dest, class_widths[0], samp_dir, mask_dir, sample,
                        prefix, id_offset)
            result += 1
    else:
        processed = [0] * class_counts[0]
        for i, dir_name in enumerate(sorted(dirs)):
            class_id = i % class_counts[0]
            new_prefix = prefix + format(class_id, '0'+str(class_widths[0])+'d')
            new_offset = id_offset + processed[class_id]
            new_source = os.path.join(source, dir_name)
            linked = link_samples(new_source, dest, class_widths[1:], class_counts[1:],
                                  samp_dir, mask_dir, new_prefix, new_offset)
            processed[class_id] += linked
        result = sum(processed)
    return result

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('source', help='source directory')
parser.add_argument('destination', help='destination directory')
parser.add_argument('samp_dir', help='samples directory')
parser.add_argument('mask_dir', help='masks directory')
parser.add_argument('class_widths', help='class width or comma-separated list of class widths')
parser.add_argument('--class_counts', '-C', 
                    help='class cardinalities per level (should be shorter than class_widths by 1)')
parser.add_argument('--prefix', '-P', help='symlink prefix')
parser.add_argument('--id_offset', '-O', help='offset to add to numeric sample ID')
args = parser.parse_args()
print('source:', args.source, 'destination:', args.destination, 'samp_dir:', args.samp_dir)
print('mask_dir:', args.mask_dir, 'class_widths: ', args.class_widths)
print('class_counts:', args.class_counts, 'prefix:', args.prefix)
print('id_offset:', args.id_offset)
class_widths = list(map(int, args.class_widths.split(',')))
class_counts = list(map(int, args.class_counts.split(','))) if args.class_counts != None else None
prefix = args.prefix if args.prefix != None else ''
id_offset = int(args.id_offset) if args.id_offset != None else 0
link_count = link_samples(args.source, args.destination, class_widths, class_counts,
                          args.samp_dir, args.mask_dir, prefix, id_offset)
print(link_count, 'samples linked')
