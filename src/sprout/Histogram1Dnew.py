import numpy as np
from numpy import ma

class Histogram1D:
    def __init__(self, bin_edges, heights, stat_errors=None, sys_errors=None, is_log_binned=False):
        """
        Initialize the histogram with bin edges, heights, and uncertainties.
        
        Parameters:
        -----------
        bin_edges : array-like
            Edges of the bins
        heights : array-like
            Height of each bin
        stat_errors : array-like, optional
            Statistical uncertainties for each bin
        sys_errors : array-like, optional
            Systematic uncertainties for each bin
        is_log_binned : bool
            Whether the histogram is logarithmically binned
        """
        self.bin_edges = np.array(bin_edges)
        
        # Convert heights to masked array, masking zeros or negative values
        self.heights = ma.masked_less_equal(np.array(heights), 0)
        
        # If no errors provided, initialize with zeros
        if stat_errors is None:
            stat_errors = np.zeros_like(heights)
        if sys_errors is None:
            sys_errors = np.zeros_like(heights)
            
        # Mask errors where heights are masked
        self.stat_errors = ma.array(stat_errors, mask=self.heights.mask)
        self.sys_errors = ma.array(sys_errors, mask=self.heights.mask)
        self.is_log_binned = is_log_binned
        
        # Calculate total errors
        self.total_errors = ma.sqrt(self.stat_errors**2 + self.sys_errors**2)
        
    @property
    def bin_centers(self):
        """Calculate bin centers based on binning type."""
        if self.is_log_binned:
            return np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        else:
            return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
            
    def __add__(self, other):
        """Add two histograms or add a constant to histogram."""
        if isinstance(other, Histogram1D):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Histograms must have the same bin edges")
                
            new_heights = self.heights + other.heights
            new_stat_errors = ma.sqrt(self.stat_errors**2 + other.stat_errors**2)
            new_sys_errors = ma.sqrt(self.sys_errors**2 + other.sys_errors**2)
            
            return Histogram1D(self.bin_edges, new_heights, new_stat_errors, 
                             new_sys_errors, self.is_log_binned)
        else:
            # Adding a constant doesn't affect uncertainties
            return Histogram1D(self.bin_edges, self.heights + other, 
                             self.stat_errors, self.sys_errors, self.is_log_binned)
            
    def __sub__(self, other):
        """Subtract two histograms or subtract a constant from histogram."""
        if isinstance(other, Histogram1D):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Histograms must have the same bin edges")
                
            new_heights = self.heights - other.heights
            new_stat_errors = ma.sqrt(self.stat_errors**2 + other.stat_errors**2)
            new_sys_errors = ma.sqrt(self.sys_errors**2 + other.sys_errors**2)
            
            return Histogram1D(self.bin_edges, new_heights, new_stat_errors, 
                             new_sys_errors, self.is_log_binned)
        else:
            # Subtracting a constant doesn't affect uncertainties
            return Histogram1D(self.bin_edges, self.heights - other, 
                             self.stat_errors, self.sys_errors, self.is_log_binned)
            
    def __mul__(self, other):
        """Multiply two histograms or multiply by a constant."""
        if isinstance(other, Histogram1D):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Histograms must have the same bin edges")
                
            new_heights = self.heights * other.heights
            
            # Error propagation for multiplication
            rel_stat_error = ma.sqrt((self.stat_errors/self.heights)**2 + 
                                   (other.stat_errors/other.heights)**2)
            rel_sys_error = ma.sqrt((self.sys_errors/self.heights)**2 + 
                                  (other.sys_errors/other.heights)**2)
            
            new_stat_errors = new_heights * rel_stat_error
            new_sys_errors = new_heights * rel_sys_error
            
            return Histogram1D(self.bin_edges, new_heights, new_stat_errors, 
                             new_sys_errors, self.is_log_binned)
        else:
            # Multiply by constant: errors scale with constant
            return Histogram1D(self.bin_edges, self.heights * other, 
                             self.stat_errors * abs(other), 
                             self.sys_errors * abs(other), 
                             self.is_log_binned)
            
    def __truediv__(self, other):
        """Divide two histograms or divide by a constant."""
        if isinstance(other, Histogram1D):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Histograms must have the same bin edges")
                
            new_heights = self.heights / other.heights
            
            # Error propagation for division
            rel_stat_error = ma.sqrt((self.stat_errors/self.heights)**2 + 
                                   (other.stat_errors/other.heights)**2)
            rel_sys_error = ma.sqrt((self.sys_errors/self.heights)**2 + 
                                  (other.sys_errors/other.heights)**2)
            
            new_stat_errors = new_heights * rel_stat_error
            new_sys_errors = new_heights * rel_sys_error
            
            return Histogram1D(self.bin_edges, new_heights, new_stat_errors, 
                             new_sys_errors, self.is_log_binned)
        else:
            # Divide by constant: errors scale with 1/constant
            return Histogram1D(self.bin_edges, self.heights / other, 
                             self.stat_errors / abs(other), 
                             self.sys_errors / abs(other), 
                             self.is_log_binned)
    
    # Support reverse operations
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    
    def __str__(self):
        """String representation of the histogram."""
        return f"Histogram1D with {len(self.heights)} bins ({np.sum(~self.heights.mask)} non-empty)"
    
    @property
    def valid_bins(self):
        """Return a boolean array indicating which bins are valid (non-empty)."""
        return ~self.heights.mask
    
    def get_valid_data(self):
        """
        Return only the valid (non-empty) bin data.
        
        Returns:
        --------
        dict containing heights, errors, and bin edges for valid bins only
        """
        valid = self.valid_bins
        return {
            'bin_edges': np.concatenate([self.bin_edges[:-1][valid], [self.bin_edges[-1]]]),
            'heights': self.heights[valid],
            'stat_errors': self.stat_errors[valid],
            'sys_errors': self.sys_errors[valid],
            'total_errors': self.total_errors[valid],
            'bin_centers': self.bin_centers[valid]
        }