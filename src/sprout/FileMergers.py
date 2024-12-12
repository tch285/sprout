#!/usr/bin/env python3

import ROOT
import os
import yaml
from tqdm import trange, tqdm
import time
import sys
import re

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

class FileMerger(object):
    def __init__(self, jobids, root_filename = "AnalysisResults.root"):
        if isinstance(jobids, int):
            self.jobids = [jobids]
        elif isinstance(jobids, list):
            self.jobids = jobids
        else:
            raise TypeError(f"Jobids parameter is not the right type; is type {type(jobids)}")
        
        output = '_'.join([str(jobid) for jobid in self.jobids])
        self.root_filename = root_filename
        self.output_dir = f"/global/cfs/projectdirs/alice/mhwang/cEEC/results/{output}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _merge_files_pattern(self, pattern, output_file_suffix):
        # code = os.system(f"hadd -v 99 -k -f {self.output_dir}/{output_file_suffix} {pattern}")
        print(f"hadd -v 99 -k -f {self.output_dir}/{output_file_suffix} {pattern}")
        # print(code)
    
    def _merge_files_split(self, filelist, output_file_suffix, split = 1000):
        # filelist_str = " ".join(filelist[:1750])
        nfiles = len(filelist)
        # print(f"Merging {nfiles} files.")
        niterations = nfiles // split + 1

        if split > nfiles:
            print(f"Merging {nfiles} with 1 iteration.")
        else:
            print(f"Merging {nfiles} with {niterations} iterations, split {split}.")
        print(len(filelist[0:1000]))
        for iteration in range(1, niterations + 1):
            ifile_min = split * (iteration - 1)
            ifile_max = split * iteration if split * iteration < nfiles else nfiles
            filelist_str = " ".join(filelist[ifile_min:ifile_max])
            
            start = time.perf_counter()
            cmd = os.system(f"hadd -v 99 -j $(nproc) -k -f {self.output_dir}/submerge_{iteration}.root {filelist_str}")
            exit_code = os.WEXITSTATUS(cmd)
            print(f"Merging (iteration {iteration} / {niterations}) completed with exit code {exit_code} in {time.perf_counter() - start:.3f} seconds.")
        
        if niterations > 1:
            cmd = os.system(f"hadd -v 99 -j $(nproc) -k -f {self.output_dir}/{output_file_suffix} {self.output_dir}/submerge_*.root")
        else:
            print("Found only one iteration, renaming...")
            os.rename(f"{self.output_dir}/submerge_1.root", f"{self.output_dir}/{output_file_suffix}")


    def _merge_files_manual_split(self, filelist, output_file_suffix, iteration, split = 300):
        # filelist_str = " ".join(filelist[:1750])
        start = time.perf_counter()
        nfiles = len(filelist)
        print(f"Merging {nfiles} files.")

        # print(f"Merging {split} files at a time...")
        niterations = nfiles // split + 1
        names = self._get_hist_names(filelist[0])
        # print(len(filelist[0:split]))
        # for iteration in range(1, niterations + 1):
        ifile_min = split * (iteration - 1)
        ifile_max = split * iteration if split * iteration < nfiles else nfiles
        if iteration >= 1 and iteration <= niterations:
            print(f'Merging files from index {ifile_min} to {ifile_max - 1}.')
            subfilelist = filelist[ifile_min:ifile_max]
            
            itr_start = time.perf_counter()
            print(f"Starting manual addition, iteration {iteration}.")
            if os.path.isfile(f"{self.output_dir}/submerge_{iteration}.root"):
                print("File already exists, skipping this iteration.")
            else:
                hists = self._manual_add_histograms(names, subfilelist)
                print(f"Starting write, iteration {iteration}.")
                self._write_histograms(hists, f"{self.output_dir}/submerge_{iteration}.root")
                print(f'Completed iteration {iteration} in {time.perf_counter() - itr_start:.2f} s.')
        elif iteration >= niterations + 1:
            # new_iteration = iteration - niterations
            # ifile_min = split * (iteration - 1)
            # ifile_max = split * iteration if split * iteration < nfiles else nfiles
            print('Placeholder for final merge, skipping and exiting 15.')
            # submerge_list = [f"{self.output_dir}/submerge_{iteration}.root" for iteration in range(1, niterations + 1)]
            # hists = self._manual_add_histograms(names, submerge_list)
            # self._write_histograms(hists, f"{self.output_dir}/{output_file_suffix}")
            # print(f'Completed full manual merge in {time.perf_counter() - start:.2f}s.')
            sys.exit(15)
        else:
            print("This iteration not needed, final merge already complete.")
            sys.exit(15)
        
    def _merge_files(self, filelist, output_file_suffix):
        # filelist_str = " ".join(filelist[:1740])
        filelist_str = " ".join(filelist)
        # LIMIT FOR HADD SEEMS TO BE 1740 FILES?
        print(f"Merging {len(filelist)} files.")
        # with open(f'{self.output_dir}/files_to_merge.txt', 'w') as f:
        #     for file in filelist:
        #         f.write(f"{file}\n")
        # os.system(f"hadd -v 0 -f {self.output_dir}/{output_file_suffix} {filelist_str}")
        # cmd = os.system(f"hadd -v 99 -k -f {self.output_dir}/{output_file_suffix} {filelist_str}")
        # cmd = os.system(f"hadd -v 99 -j 300 -k -f {self.output_dir}/{output_file_suffix} @{self.output_dir}/files_to_merge.txt")
        # cmd = os.system(f"hadd -v 99 -j $(nproc) -k -f {self.output_dir}/{output_file_suffix} @{self.output_dir}/files_to_merge.txt")
        cmd = os.system(f"hadd -v 99 -j $(nproc) -k -f {self.output_dir}/{output_file_suffix} {filelist_str}")
        # cmd = os.system(f"hadd -v 99 -k -f {self.output_dir}/{output_file_suffix} @{self.output_dir}/files_to_merge.txt")
        exit_code = os.WEXITSTATUS(cmd)
        print(f"Merging completed with exit code {exit_code}")
        # print(f"hadd -v 99 -k -f {self.output_dir}/{output_file_suffix} {filelist_str}")
        # print(f"hadd -v 0 -f {self.output_dir}/{output_file_suffix}")
        # self._build_aggregate_histograms(filelist[0])
        # for filename in tqdm(filelist):
        # # for filename in filelist:
        #     # print(filename)
        #     try:
        #         with ROOT.TFile(filename, "READ") as inf:
        #             for name in self.hist_names:
        #                 # print(name)
        #                 self.histograms[name].Add(inf.Get(name))
        #     except OSError:
        #         print(filename)
        #     except TypeError as e:
        #         print(name)
        #         print(filename)
        #         # raise TypeError("problem!")
        #         print(e)
        # with ROOT.TFile(f"{self.output_dir}/{output_file_suffix}", "RECREATE") as outf:
        #     for hist in self.histograms.values():
        #         outf.WriteTObject(hist)

    def _build_aggregate_histograms(self, file):
        self.histograms = {}
        kwds = ['E2C', 'jet_pt', 'Nevents', 'JES', 'JetMatchingQA']
        with ROOT.TFile(file, "READ") as testf:
            titer = ROOT.TIter(testf.GetListOfKeys())
            self.hist_names = [key.GetName() for key in titer]
            self.hist_names = [name for name in self.hist_names if any(kw in name for kw in kwds)]
            for name in self.hist_names:
                newhist = testf.Get(name).Clone()
                newhist.Reset("ICESM")
                newhist.SetDirectory(0)
                self.histograms[name] = newhist
                # except 

    def _get_hist_names(self, file):
        # kwds = ['E2C', 'jet_pt', 'Nevents', 'JES', 'JetMatchingQA']
        with ROOT.TFile(file, "READ") as testf:
            titer = ROOT.TIter(testf.GetListOfKeys())
            names = [key.GetName() for key in titer]
            # self.hist_names = [name for name in self.hist_names if any(kw in name for kw in kwds)]
        return names

    def _get_init_histograms(self, file, names):
        histograms = {}
        # kwds = ['E2C', 'jet_pt', 'Nevents', 'JES', 'JetMatchingQA']
        with ROOT.TFile(file, "READ") as f:
            for name in names:
                newhist = f.Get(name).Clone()
                # newhist = f.Get(name)
                # newhist.Reset("ICESM")
                # RooUnfoldResponse doesn't have SetDirectory (no need)
                if isinstance(newhist, (ROOT.TH1)):
                    newhist.SetDirectory(0)
                histograms[name] = newhist
        return histograms
    
    def _manual_add_histograms(self, names, filelist):
        histograms = self._get_init_histograms(filelist[0], names)

        for filename in tqdm(filelist[1:]):
            # print(filename)
            with ROOT.TFile(filename, "READ") as f:
                for name in names:
                    # print(name)
                    histograms[name].Add(f.Get(name))
        return histograms

    def _write_histograms(self, hists, ofname):
        with ROOT.TFile(ofname, "RECREATE") as outf:
            for name in hists.keys():
                outf.WriteTObject(hists[name])
    
    def merge(self):
        raise NotImplementedError()

class MCFileMerger(FileMerger):
    def __init__(self, jobids, root_filename = "AnalysisResults.root", xsec_file = "", num_pThat_bins = 20, is_fastsim = False):
        super(MCFileMerger, self).__init__(jobids, root_filename)
        self.xsec_file = xsec_file
        print(f"Using xsec file {self.xsec_file}...")
        self.num_pThat_bins = num_pThat_bins
        self.is_fastsim = is_fastsim
        self.subfiles_dir = f"{self.output_dir}/subfiles"
    
    def merge(self, output_filename = "final_merged_results.root", unscaled_file_tmpl = "merged_{}.root", scaled_file_tmpl = "scaled_{}.root"):
        self.unscaled_file_tmpl = unscaled_file_tmpl
        self.scaled_file_tmpl = scaled_file_tmpl
        self.output_filename = output_filename


        unsc_start = time.perf_counter()
        print("Beginning merge of unscaled files.")
        self._merge_unscaled()
        unsc_dur = time.perf_counter() - unsc_start
        print(f"Completed unscaled merge in {unsc_dur:.2f} seconds.")
        
        scaling_start = time.perf_counter()
        print("Beginning scaling of merged files.")
        self._scale_files()
        scaling_dur = time.perf_counter() - scaling_start
        print(f"Completed scaling in {scaling_dur:.2f} seconds.")

        sc_start = time.perf_counter()
        print("Beginning merge of scaled files.")
        self._merge_scaled()
        sc_dur = time.perf_counter() - sc_start
        print(f"Completed scaled merge in {sc_dur:.2f} seconds.")

        full_dur = time.perf_counter() - unsc_start

        print( "Completed full merging. Summary:")
        print(f"    unscaled merge in {unsc_dur:.2f} seconds.")
        print(f"    scaling in {scaling_dur:.2f} seconds.")
        print(f"    scaled merge in {sc_dur:.2f} seconds.")
        print(f"Full duration: {full_dur:.2f} seconds.")

    def _find_templates(self): # find templates for specific bin
        templates = []
        if self.is_fastsim:
            for jobid in self.jobids:
                for root_dir, subdirs, files in os.walk(f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}"):
                    if not subdirs and len(files) == 1 and files[0] == self.root_filename:
                        testname = f"{root_dir}/{self.root_filename}"
                        print(f"Found testname: {testname}")
                        break
                
                match = re.fullmatch(f"^/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}/([0-9/]+)/(1[0-9]|20|[1-9])/([0-9]*)/{self.root_filename}", testname)
                if not match:
                    print("Couldn't find match to test name, exiting.")
                    sys.exit(1)
                subdirs, ptbin, subin = match.groups()
                template = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}/{subdirs}/{{}}"
                print(f"Found template. Example with bin 1 would be {template.format(1)}")
                templates.append(template)
        else:
            for jobid in self.jobids:
                if jobid <= 31852882:
                    train_id = 798
                elif jobid > 31852882 and jobid < 33081322:
                    train_id = 804
                else:
                    train_id = 806
                
                if jobid < 33081322:
                    parent_dir = "LHC18b8_charge"
                    dataset = "LHC18b8"
                    for trigger in ["cent_woSDD", "fast"]:
                        templ = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}/{parent_dir}/{train_id}/{dataset}_{trigger}/{{}}"
                        templates.append(templ)
                else:
                    parent_dir = "LHC23a3"
                    dataset = "LHC23a3"
                    for trigger in ["b", "c"]:
                        templ = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}/{parent_dir}/{train_id}/{dataset}{trigger}/{{}}"
                        templates.append(templ)
        
        return templates

    def _merge_unscaled(self):
        if not os.path.exists(self.subfiles_dir):
            os.makedirs(self.subfiles_dir)

        templates = self._find_templates()

        print(f"Merging {self.num_pThat_bins} bins...")
        for bin in trange(1, self.num_pThat_bins + 1):
            filelist = []
            for templ in templates:
                base_dir = templ.format(bin)
                print(base_dir)
                for root_dir, subdirs, files in os.walk(base_dir):
                    if not subdirs:
                        for file in files:
                            if file == self.root_filename:
                                filelist.append(f"{root_dir}/{self.root_filename}")
            # for jobid in self.jobids:
            #     for trigger in ["cent_woSDD", "fast"]:
            #         base_dir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}/LHC18b8_charge/798/LHC18b8_{trigger}/{bin}"
            #         for root_dir, subdirs, _ in os.walk(base_dir):
            #             if not subdirs:
            #                 filelist.append(f"{root_dir}/{self.root_filename}")
            self._merge_files(filelist, f"subfiles/{self.unscaled_file_tmpl.format(bin)}")

    def _scale_files(self):
        scaler = Scaler()
        inf_tmpl = self.subfiles_dir + "/" + self.unscaled_file_tmpl
        outf_tmpl = self.subfiles_dir + "/" + self.scaled_file_tmpl
        scaler.scale_histograms(xsec_file = self.xsec_file, inf_tmpl = inf_tmpl, num_pThat_bins = self.num_pThat_bins, outf_tmpl = outf_tmpl)

    def _merge_scaled(self):
        output_dir = self.subfiles_dir + "/" + self.scaled_file_tmpl
        filelist = [output_dir.format(bin) for bin in range(1, self.num_pThat_bins + 1)]
        self._merge_files(filelist, self.output_filename)


class DataFileMerger(FileMerger):
    def __init__(self, jobids, root_filename = "AnalysisResults.root"):
        super(DataFileMerger, self).__init__(jobids, root_filename)

    def merge(self, output_filename = "merged_results.root"):
        filelist = []
        for jobid in self.jobids:
            base_dir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}"
            for root_dir, subdirs, files in os.walk(base_dir):
                if not subdirs:
                    # [file for file in files if file == self.root_filename]
                    for file in files:
                        if file == self.root_filename:
                            filelist.append(f"{root_dir}/{self.root_filename}")
        
        # print(filelist[-1])
        # print(len(filelist))
        # self._merge_files_pattern(f"{base_dir}/*/AnalysisResults.root", output_filename)
        start = time.perf_counter()
        # self._merge_files(filelist, output_filename)
        self._merge_files_split(filelist, output_filename)
        print(f"Completed merge in {time.perf_counter() - start} seconds.")

class MCManualFileMerger(FileMerger):
    def __init__(self, jobids, iteration, root_filename = "AnalysisResults.root", split = 300):
        super(MCManualFileMerger, self).__init__(jobids, root_filename)
        self.iteration = iteration
        self.split = split

    def merge(self, output_filename = "merged_results.root"):
        filelist = []
        for jobid in self.jobids:
            base_dir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang/{jobid}"
            for root_dir, subdirs, files in os.walk(base_dir):
                subdirs.sort()
                files.sort()
                if not subdirs:
                    # [file for file in files if file == self.root_filename]
                    for file in files:
                        if file == self.root_filename:
                            filelist.append(f"{root_dir}/{self.root_filename}")
        
        # print(filelist[-1])
        # print(len(filelist))
        # self._merge_files_pattern(f"{base_dir}/*/AnalysisResults.root", output_filename)
        start = time.perf_counter()
        # self._merge_files(filelist, output_filename)
        self._merge_files_manual_split(filelist, output_filename, self.iteration, split = self.split)
        print(f"Completed merge in {time.perf_counter() - start} seconds.")



# Macro to scale histograms of all Pt-hard bins, using xsec from a yaml file.
# This script expects files X/AnalysisResults.root, and will output scaled histograms
# to the same file, in a new output list with suffix "Scaled". The script will automatically loop over
# all output lists, subject to some simple criteria that covers basic use cases (can be adapted as needed).
#
# There is an option "bRemoveOutliers" to remove outliers from certain histograms. The features are
# currently hard-coded below so you will need to modify the code as needed. This feature is adapted from code of Raymond Ehlers.
# 
# Modifications on original script from James to use as class
# 
# Authors: Tucker Hwang (tucker_hwang@berkeley.edu), James Mulligan (james.mulligan@berkeley.edu)
#

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

class Scaler(object):
    def __init__(self):
        pass
        # self.jobid = jobid

###################################################################################
# Main function
    def scale_histograms(self, xsec_file, inf_tmpl, num_pThat_bins, outf_tmpl):
        # Option to remove outliers from specified histograms
        # If the average bin content stays below the "outlierLimit" for "outlierNBinsThreshold" bins, it is removed
        bRemoveOutliers = False
        outlierLimit = 2
        outlierNBinsThreshold=4
        
        # problem_bins = [4, 6, 8, 9, 10, 19]
        # prefix = f"/global/cfs/cdirs/alice/mhwang/cEEC/results/{self.jobid}"
        
        # Option to print out detailed info about scaling and outlier removal
        verbose = False
        
        # Read the cross-section, and scale histograms
        num_pThat_bins = 20
        with open(xsec_file, 'r') as stream:
            xsecs = yaml.safe_load(stream)

        # Compute average number of events per bin
        nEventsSum = 0
        for bin in range(1, num_pThat_bins+1):
            with ROOT.TFile(inf_tmpl.format(bin), "READ") as f:
                hNevents = f.Get("hNevents")
                nEvents = hNevents.GetBinContent(2)
                nEventsSum += nEvents
                # titer = ROOT.TIter(f.GetListOfKeys())
                # keys = [key for key in titer]
        nEventsAvg = nEventsSum*1./num_pThat_bins
        print(f"Average events per pT-hat bin: {nEventsAvg}")

        for bin in range(1, num_pThat_bins+1):
            print(f"    Scaling pT-hat bin {bin} of {num_pThat_bins}")
            # inf_name = f"{prefix}/merged_files_corr_{bin}.root" if bin in problem_bins else f"{prefix}/merged_files_{bin}.root"
            # outf_name = f"{prefix}/scaled_merged_files_corr_{bin}.root" if bin in problem_bins else f"{prefix}/scaled_merged_files_{bin}.root"
            inf_name = inf_tmpl.format(bin)
            outf_name = outf_tmpl.format(bin)

            with ROOT.TFile(inf_name, "READ") as fin, ROOT.TFile(outf_name, "recreate") as fout:
                xsec = xsecs[bin]
                nEvents = fin.Get("hNevents").GetBinContent(2)
                eventScaleFactor = nEvents / nEventsAvg
                scaleFactor = xsec / eventScaleFactor
                print(f"    eventScaleFactor: {eventScaleFactor}")
                print(f"    total scaleFactor: {scaleFactor}")

                # Now, scale all the histograms
                keys = ROOT.TIter(fin.GetListOfKeys())
                # print(keys)
                for key in keys:
                    name = key.GetName()
                    # print(name)
                    if "Scaled" in name:
                        continue
                    if "roounfold" in name:
                        continue
                    obj = fin.Get(name)
                    if obj:
                        self._scale_all_histograms(obj, scaleFactor, verbose, bRemoveOutliers, outlierLimit, outlierNBinsThreshold, bin-1, num_pThat_bins, name)
                    else:
                        print('obj not found!')
            
                    # obj.Write("%sScaled" % obj.GetName())
                    # obj.SetName(f"{name}")
                    fout.WriteTObject(obj)
                    
                    # if remove_unscaled:
                    # 	f.Delete('{};1'.format(obj.GetName()))  # Remove unscaled histogram

###################################################################################
# Function to iterate recursively through an object to scale all TH1/TH2/THnSparse
    def _scale_all_histograms(self, obj, scaleFactor, verbose, bRemoveOutliers=False, limit=2, nBinsThreshold=4, pTHardBin=0, num_pThat_bins=20, taskName=""):
        
        # Set Sumw2 if not already done
        if obj.InheritsFrom(ROOT.THnBase.Class()):
            # NOTE: checking == 0 doesn't work here. Will be using != 0 to force it to sumw2 
            if obj.GetSumw2() != 0:
                obj.Sumw2()
                if verbose:
                    print('Set Sumw2 on {}'.format(obj.GetName()))
        else:
            if obj.GetSumw2N() != 0:
                obj.Sumw2()
                if verbose:
                    print('Set Sumw2 on {}'.format(obj.GetName()))
        
        if obj.InheritsFrom(ROOT.TProfile.Class()):
            if verbose:
                print("TProfile %s not scaled..." % obj.GetName())
        elif obj.InheritsFrom(ROOT.TH2.Class()):
            obj.Scale(scaleFactor)
            if verbose:
                print("TH2 %s was scaled..." % obj.GetName())
        elif obj.InheritsFrom(ROOT.TH1.Class()):
            # if bRemoveOutliers:
            # 	name = obj.GetName()
                #only perform outlier removal on these couple histograms
                # if "Pt" in name:
                # 	removeOutliers(pTHardBin, num_pThat_bins, obj, verbose, limit, nBinsThreshold, 1, taskName)
            obj.Scale(scaleFactor)
            if verbose:
                print("TH1 %s was scaled..." % obj.GetName())
        elif obj.InheritsFrom(ROOT.THnBase.Class()):
            obj.Scale(scaleFactor)
            if verbose:
                print("THnSparse %s was scaled..." % obj.GetName())
        else:
            if verbose:
                print("Not a histogram!")
                print(obj.GetName())
            for subobj in obj:
                self._ScaleAllHistograms(subobj, scaleFactor, verbose, bRemoveOutliers, limit, nBinsThreshold, pTHardBin, taskName)

###################################################################################
# Function to remove outliers from a TH3 (i.e. truncate the spectrum), based on projecting to the y-axis
# It truncates the 3D histogram based on when the 1D projection 4-bin moving average has been above
# "limit" for "nBinsThreshold" bins.
# def removeOutliers(pTHardBin, num_pThat_bins, hist, verbose, limit=2, nBinsThreshold=4, dimension=3, taskName=""):

# 	#Project to the pT Truth axis
# 	if dimension==3:
# 		histToCheck = hist.ProjectionY("{}_projBefore".format(hist.GetName()))
# 	if dimension==2:
# 		histToCheck = hist.ProjectionX("{}_projBefore".format(hist.GetName()))
# 	if dimension==1:
# 		histToCheck = hist

# 	# Check with moving average
# 	foundAboveLimit = False
# 	cutLimitReached = False
# 	# The cut index is where we decided cut on that row
# 	cutIndex = -1
# 	nBinsBelowLimitAfterLimit = 0
# 	# nBinsThreshold= n bins that are below threshold before all bins are cut

# 	if verbose:
# 		(preMean, preMedian) = GetHistMeanAndMedian(histToCheck)
        
# 	for index in range(0, histToCheck.GetNcells()):
# 		if verbose:
# 			print("---------")
# 		avg = MovingAverage(histToCheck, index = index, numberOfCountsBelowIndex = 2, numberOfCountsAboveIndex = 2)
# 		if verbose:
# 			print("Index: {0}, Avg: {1}, BinContent: {5}, foundAboveLimit: {2}, cutIndex: {3}, cutLimitReached: {4}".format(index, avg, foundAboveLimit, cutIndex, cutLimitReached, histToCheck.GetBinContent(index)))
# 		if avg > limit:
# 			foundAboveLimit = True
                
# 		if not cutLimitReached:
# 			if foundAboveLimit and avg <= limit:
# 				if cutIndex == -1:
# 					cutIndex = index
# 				nBinsBelowLimitAfterLimit += 1
                    
# 			if nBinsBelowLimitAfterLimit != 0 and avg > limit:
# 				# Reset
# 				cutIndex = -1
# 				nBinsBelowLimitAfterLimit = 0

# 			if nBinsBelowLimitAfterLimit > nBinsThreshold:
# 				cutLimitReached = True
# 				break #no need to continue the loop - we found our cut index

# 	# Do not perform removal here because then we miss values between the avg going below
# 	# the limit and crossing the nBinsThreshold
# 	if verbose:
# 		print("Hist checked: {0}, cut index: {1}".format(histToCheck.GetName(), cutIndex))
    
# 	# Use on both TH1 and TH2 since we don't start removing immediately, but instead only after the limit
# 	if cutLimitReached:
# 		if verbose:
# 			print("--> --> --> Removing outliers")
# 		# Check for values above which they should be removed by translating the global index
# 		x = ctypes.c_int(0)
# 		y = ctypes.c_int(0)
# 		z = ctypes.c_int(0)
# 		for index in range(0, hist.GetNcells()):
# 			# Get the bin x, y, z from the global bin
# 			hist.GetBinXYZ(index, x, y, z)
# 			if dimension==3:
# 				if y.value >= cutIndex:
# 					if hist.GetBinContent(index) > 1e-3:
# 						if verbose:
# 							print("Cutting for index {}. y bin {}. Cut index: {}".format(index, y, cutIndex))
# 						hist.SetBinContent(index, 0)
# 						hist.SetBinError(index, 0)
# 			if dimension==2:
# 				#for the response matrix the pT Truth is on the y-Axis
# 				if hist.GetName()=="hResponseMatrixEMCal":
# 					x.value=y.value
# 				if x.value >= cutIndex:
# 					if hist.GetBinContent(index) > 1e-3:
# 						if verbose:
# 							print("Cutting for index {}. x bin {}. Cut index: {}".format(index, x, cutIndex))
# 						hist.SetBinContent(index, 0)
# 						hist.SetBinError(index, 0)
# 			if dimension==1:
# 				if x.value >= cutIndex:
# 					if hist.GetBinContent(index) > 1e-3:
# 						if verbose:
# 							print("Cutting for index {}. x bin {}. Cut index: {}".format(index, x, cutIndex))
# 						hist.SetBinContent(index, 0)
# 						hist.SetBinError(index, 0)
                    
# 	else:
# 		if verbose:
# 			print("Hist {} did not have any outliers to cut".format(hist.GetName()))

# 	# Check the mean and median
# 	# Use another temporary hist
# 	if dimension==3:
# 		histToCheckAfter = hist.ProjectionY()
# 	if dimension==2:
# 		if hist.GetName()=="hResponseMatrixEMCal":
# 			histToCheckAfter = hist.ProjectionY()
# 	else:
# 			histToCheckAfter = hist.ProjectionX()
# 	if dimension==1:
# 		histToCheckAfter = hist

# 	if verbose:
# 		(postMean, postMedian) = GetHistMeanAndMedian(histToCheckAfter)
# 		print("Pre  outliers removal mean: {}, median: {}".format(preMean, preMedian))
# 		print("Post outliers removal mean: {}, median: {}".format(postMean, postMedian))
# 	outlierFilename = "{}OutlierRemoval_{}.pdf".format(hist.GetName())
# 	if "Pt" in hist.GetName():
# 		plotOutlierPDF(histToCheck, histToCheckAfter, pTHardBin, num_pThat_bins, outlierFilename, verbose, "hist E", True)

########################################################################################################
# def GetHistMeanAndMedian(hist):
# 	# Median
# 	# See: https://root-forum.cern.ch/t/median-of-histogram/7626/5
# 	x = ctypes.c_double(0)
# 	q = ctypes.c_double(0.5)
# 	# Apparently needed to be safe(?)
# 	hist.ComputeIntegral()
# 	hist.GetQuantiles(1, x, q)
        
# 	mean = hist.GetMean()
# 	return (mean, x.value)

########################################################################################################
# def MovingAverage(hist, index, numberOfCountsBelowIndex = 0, numberOfCountsAboveIndex = 2):
# 	"""
# 	# [-2, 2] includes -2, -1, 0, 1, 2
# 	"""
# 	# Check inputs
# 	if numberOfCountsBelowIndex < 0 or numberOfCountsAboveIndex < 0:
# 		print("Moving average number of counts above or below must be >= 0. Please check the values!")
                    
# 	count = 0.
# 	average = 0.
# 	for i in range(index - numberOfCountsBelowIndex, index + numberOfCountsAboveIndex + 1):
# 		# Avoid going over histogram limits
# 		if i < 0 or i >= hist.GetNcells():
# 			continue
# 		#print("Adding {}".format(hist.GetBinContent(i)))
# 		average += hist.GetBinContent(i)
# 		count += 1
        
# 	#if count != (numberOfCountsBelowIndex + numberOfCountsAboveIndex + 1):
# 	#    print("Count: {}, summed: {}".format(count, (numberOfCountsBelowIndex + numberOfCountsAboveIndex + 1)))
# 	#exit(0)

# 	return average / count

########################################################################################################
# Plot basic histogram    ##############################################################################
########################################################################################################
# def plotOutlierPDF(h, hAfter, pTHardBin, num_pThat_bins, outputFilename, verbose, drawOptions = "", setLogy = False):
    
# 	c = ROOT.TCanvas("c","c: hist",600,450)
# 	c.cd()
# 	if setLogy:
# 		c.SetLogy()
# 	h.GetXaxis().SetRangeUser(0,250)
# 	h.Draw("hist")
# 	h.SetLineColor(616)
# 	hAfter.SetLineColor(820)
# 	hAfter.Draw("same hist")
    
# 	leg1 = ROOT.TLegend(0.17,0.7,0.83,0.85,"outlier removal of Bin {}".format(pTHardBin))
# 	leg1.SetFillColor(10)
# 	leg1.SetBorderSize(0)
# 	leg1.SetFillStyle(0)
# 	leg1.SetTextSize(0.04)
# 	leg1.AddEntry(h, "before", "l")
# 	leg1.AddEntry(hAfter, "after", "l")
# 	leg1.Draw("same")

# 	c.Print("{}".format(outputFilename))
# 	'''
# 	if pTHardBin == 0: #if first pt-hard bin, open a .pdf
# 		if verbose:
# 			print("Add first pT Hard bin to pdf with name: {0}".format(outputFilename))
# 	elif pTHardBin==num_pThat_bins-1: #otherwise add pages to the file
# 		if verbose:
# 			print("Add last pT Hard bin to pdf with name: {0}".format(outputFilename))
# 		c.Print("{})".format(outputFilename))
# 	else: #otherwise close the file
# 		if verbose:
# 			print("Add further pT Hard bin to pdf with name: {0}".format(outputFilename))
# 		c.Print("{}".format(outputFilename))
# 	'''

# 	c.Close()

########################################################################################################
# Get Jet radius from list analysis label                       #############################################
########################################################################################################
# def getRadiusFromlistName(listName):

# 	radius = 0.0
# 	if "01" in listName:
# 		radius = 0.1
# 	if "02" in listName:
# 		radius = 0.2
# 	elif "03" in listName:
# 		radius = 0.3
# 	elif "04" in listName:
# 		radius = 0.4
# 	elif "05" in listName:
# 		radius = 0.5
# 	elif "06" in listName:
# 		radius = 0.6
# 	return radius

#---------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
# 	print("Executing scaleHistograms.py...")
# 	print("")
    
# 	# Define arguments
# 	parser = argparse.ArgumentParser(description='Plot analysis histograms')
# 	parser.add_argument('-c', '--configFile', action='store',
# 											type=str, metavar='configFile',
# 											default='analysis_config.yaml',
# 											help="Path of config file for jetscape analysis")
# 	parser.add_argument('-r', '--remove_unscaled', action='store_false',
# 											help='Remove unscaled histograms')
    
# 	# Parse the arguments
# 	args = parser.parse_args()
    
# 	print('Configuring...')
# 	print('configFile: \'{0}\''.format(args.configFile))
# 	print('----------------------------------------------------------------')
    
# 	# If invalid configFile is given, exit
# 	if not os.path.exists(args.configFile):
# 		print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
# 		sys.exit(0)

# 	scaleHistograms(configFile = args.configFile, remove_unscaled = args.remove_unscaled)