"""
Class to aid in the calculation of the emission surface.
"""

from imgcube.imagecube import imagecube
from scipy.interpolate import interp1d
from detect_peaks import detect_peaks
from prettyplots.gaussianprocesses import Matern32_model
from prettyplots.prettyplots import running_variance, sort_arrays
import numpy as np
import matplotlib.pyplot as plt


class linecube:

    msun = 1.9886e30    # Solar mass in [kg].

    def __init__(self, path, orientation='east', inc=0.0, dist=1.0, x0=0.0,
                 y0=0.0, rin=None, rout=None, mstar=1.0, vlsr=0.0, dV=300.,
                 psi=0.0, downsample=False, clip_noise=True, nsigma=3, rms=0.0,
                 nchans=10, mask=None, nbeams=0.0, clip_profile=False,
                 verbose=True):
        """
        Read in the cube file using the `imagecube` class. Simple regridding
        functionality is provided to downsample the data and to clip noisy
        pixels.

        - Input -

        path:           Relative path to the fits cube to use. This must have
                        been rotated such that the major axes are aligned with
                        the x-axis.
        orientation:    Which side is the blue-shifted major axis.
        inc:            Source inclination in [deg].
        dist:           Source distance in [pc].
        x0, y0:         Source offset position in [arcsec].
        rin, rout       Inner and outer radii of the source in [au].
        mstar:          Stellar mass of the central star in [Msun].
        vlsr:           Source velocity in [m/s].
        dV:             Intrinsic linewidth in [m/s].
        psi:            Flaring angle of the emission surface in [deg].
        clip_noise:     Clip the data based on a noise threshold.
        nsigma:         Clipping threshold.
        rms:            The RMS noise in a line free channel in [K]. If not
                        provided, will attempt to estimate this using the end
                        `nchans`.
        nchans:         Number of channels to use to estimate the RMS noise.
        maks:           Path to a fits file to use for clipping.
        verbose:        Output messages describing what's going on.
        """

        # Read in the file and set up the disk orientation.
        self.path = path
        self.cube = imagecube(path, mask)
        self.inc = inc
        if self.inc == 0.0:
            raise ValueError("Disk must be inclined to observe flaring.")
        if orientation.lower() == 'east':
            self.PA = -90.
        elif orientation.lower() == 'west':
            self.PA = 90.
        else:
            raise ValueError("Orientation must be `east` or `west`.")
        self.dist = max(dist, 1.0)
        self.verbose = verbose

        # Deprojected coordinates.
        coords = self.cube._deproject_polar(inc=self.inc, PA=self.PA,
                                            x0=x0, y0=y0, dist=self.dist)
        self.rvals, self.tvals = coords

        # Radial profiles.
        if rin is not None:
            self.rin = rin
        else:
            self.rin = 0.0
        if rout is not None:
            self.rout = rout
        else:
            self.rout = self.rvals.max() / np.sqrt(3.)

        # Downsample the data to boost the signal to noise in the channels.
        if downsample:
            self.cube.downsample_cube(downsample, center=True)
        self.data = self.cube.data
        self.data = np.where(self.data < 0.0, 0.0, self.data)

        # Grab the cube axes.
        self.xaxis = self.cube.xaxis
        self.yaxis = self.cube.yaxis
        self.velax = self.cube.velax

        # Use the brightness profile to clip low value pixels.
        # This is the mean minus the standard deviation.
        if clip_profile:
            if verbose:
                print("Clipping below the brightness profile.")
            rpnts, Tb, dTb = self.brightness_profile(nbins=100)
            profile = interp1d(rpnts, Tb - 2 * dTb, fill_value='extrapolate')
            self.data = np.where(self.data >= profile(self.rvals),
                                 self.data, 0.0)

        # Use the noise to clip low signal pixels.
        if clip_noise:
            if rms > 0.0:
                self.rms = rms
            else:
                self.rms = self.estimate_rms(nchans)
            self.clip = nsigma * self.rms
            if self.verbose:
                print("Clipping data below %.3f K." % self.clip)
            self.data = np.where(abs(self.data) >= self.clip, self.data, 0.0)
        return

    def brightness_profile(self, rmin=None, rmax=None, nbins=50):
        """
        Return the brightness temperature profile in [K].

        - Input -

        rmin:       Minimum radial distance in [au] for the fitting.
        rmax:       Maximum radial distance in [au] for the fitting.
        nbins:      Number of samples across the radius.

        - Output -

        rpnts:      Radial sampling points.
        Tb:         Brightness proflie in [K]
        dTb:        Standard deviation in each bin in [K]
        """

        # Define the radial grid to use.
        if rmin is None:
            rmin = self.rin
        if rmax is None:
            rmax = self.rout
        rpnts = np.linspace(rmin, rmax, nbins)
        rbins = np.linspace(rmin - 0.5 * np.diff(rpnts)[0],
                            rmax + 0.5 * np.diff(rpnts)[0],
                            nbins + 1)

        # Bin the profile.
        Tflat = np.amax(self.data, axis=0).flatten()
        rflat = self.rvals.flatten()
        ridxs = np.digitize(rflat, rbins)

        Tb = [np.nanmean(Tflat[ridxs == r]) for r in range(1, rbins.size)]
        dTb = [np.nanstd(Tflat[ridxs == r]) for r in range(1, rbins.size)]
        return rpnts, np.squeeze(Tb), np.squeeze(dTb)

    def estimate_rms(self, N=5):
        """Estimate the noise from centre of the end N channels."""
        dx = int(self.cube.nxpix / 4)
        return np.nanstd([self.data[:N, dx:-dx, dx:-dx],
                          self.data[-N:, dx:-dx, dx:-dx]])

    def emission_surface_GP(self, x0=0.0, y0=0.0, inc=None, include_dTb=False,
                            plot=True):
        """
        Return a GP model of the emission surface. Points will be masked if
        they fall below the average radial Tb value.

        - Input -

        x0, y0:         Source centre offsets in [arcsec].
        inc:            Source inclination in [deg].
        include_dTb:    Include uncertainties in the clipping values.
        plot:           Diagnostic plots.
        """

        if inc is None:
            inc = self.inc
        coords = self.find_surface(x0=x0, y0=y0, inc=inc)

        # Only keep points with a Tb value equal to greater than the azimuthal
        # average (not taking into account flaring.).
        r, Tb, dTb = self.brightness_profile()

        fig, ax = plt.subplots()
        ax.errorbar(r, Tb, dTb, fmt='.k', capsize=0.0)

        # Make the points depending on their brightness temperature.
        minTb = np.interp(coords[0], r, Tb + include_dTb * dTb)
        mask = coords[2] >= minTb

        if plot:
            fig, ax = plt.subplots()
            ax.errorbar(coords[0], coords[1], fmt='.', color='gray', mew=0.0)
            ax.set_xlabel('Radius (au)')
            ax.set_ylabel('Height (au)')

        # Calculate the GP model using this.
        r, z = sort_arrays(coords[0][mask], coords[1][mask])
        dz = running_variance(z, window=min(50, len(z)))**0.5
        if plot:
            ax.errorbar(r, z, dz, fmt='.r', capsize=0.0, mew=0.0)
            ax.set_xlabel('Radius (au)')
            ax.set_ylabel('Height (au)')

        r, z, dz = Matern32_model(r, z, dz, return_var=True, jitter=True)
        if plot:
            ax.fill_between(r, z - dz, z + dz, color='k', alpha=0.3)
            ax.plot(r, z, c='k')

        return r, z, dz

    def emission_surface_binned(self, **kwargs):
        """
        Return radial profiles for the height and temperature of the emission
        surface. This is a wrapper of the `find_surface` and `bin_surface`
        functions. See those for a better understanding of the kwargs.
        """
        coords = self.find_surface(x0=kwargs.pop('x0', 0.0),
                                   y0=kwargs.pop('y0', 0.0),
                                   mpd=kwargs.pop('mpd', 0.0),
                                   mph=kwargs.pop('mph', 0.0),
                                   inc=kwargs.pop('inc', None))
        return self.bin_surface(coords, **kwargs)

    def bin_surface(self, coords, rmin=None, rmax=None, nbins=50,
                    weighted=True):
        """
        Return the binned coordinates.

        - Input Variables -

        coords:     Coordinates from `find_surface`.
        rbins:      The centre of the bins used for the averaging in [au]. If
                    none are specified, will try to estimate them by sampling
                    between the minimum and maximum value with `nbins` points.
        nbins:      If `rbins` is not specified, the number of bins to use.
        weighted:   Use an intensity weighted percentile.

        - Output -

        rbins:      The bin centres in [au]. Will just return those provided.
        zbins:      The [16th, 50th, 84th] percentiles of the surface height in
                    [au].
        tbins:      The [16th, 50th, 84th] percentile of the brightness
                    temperature of the bin in [K].
        """

        # Estimate the bins.
        if rmin is None:
            rmin = np.nanmin(coords[0])
        if rmax is None:
            rmax = np.nanmax(coords[0])

        rbins = np.linspace(rmin, rmax, nbins)
        width = 0.5 * np.diff(rbins)[0]
        rbins = np.linspace(rmin - width, rmax + width, nbins + 1)

        # Bin all the data and average.
        ridxs = np.digitize(coords[0], rbins)
        zbins, tbins = [], []
        for r in range(1, rbins.size):
            if weighted:
                try:
                    bin = self._weighted_percentiles(coords[1][ridxs == r],
                                                     coords[2][ridxs == r])
                except:
                    bin = np.nan
            else:
                bin = np.nanpercentile(coords[1][ridxs == r], [16, 50, 84])
            if np.isfinite(bin).all():
                zbins += [bin]
            else:
                zbins += [[np.nan, np.nan, np.nan]]
            if weighted:
                try:
                    bin = self._weighted_percentiles(coords[2][ridxs == r],
                                                     coords[2][ridxs == r])
                except:
                    bin = np.nan
            else:
                bin = np.nanpercentile(coords[2][ridxs == r], [16, 50, 84])
            if np.isfinite(bin).all():
                tbins += [bin]
            else:
                tbins += [[np.nan, np.nan, np.nan]]

        # Reshape the arrays and return.
        rbins = np.average([rbins[1:], rbins[:-1]], axis=0)
        zbins = np.squeeze(zbins).T
        tbins = np.squeeze(tbins).T
        return rbins, zbins, tbins

    def find_surface(self, x0=0.0, y0=0.0, mph=0.0, mpd=0.0, inc=None):
        """
        Follow Pinte et al. (2018) to recover emission surface.

        - Input Variables -

        x0, y0:     Coordinates [arcseconds] of the centre of the disk.
        mph:        Minimum peak height in [K]. If None, use all pixels.
        mpd:        Minimum distance between peaks in [arcseconds].
        inc:        Inclination of the disk in [degrees]. If none is given then
                    use the value provided when intialising the class. This is
                    useful to test how different inclinations will affect the
                    profiles.

        - Output -

        coords:     A [3 x N] array where N is the number of successfully found
                    ellipses. Each ellipse yields a (r, z, Tb) trio. Distances
                    are in [au] (coverted using the provided distance) and the
                    brightness temperature in [K].
        """

        coords = []

        # Convert from degrees to radians for inclination.
        if inc is None:
            inc = self.inc
        inc = np.radians(inc)

        for c, channel in enumerate(self.data):

            # Avoid empty channels.
            if np.nanmax(channel) < mph:
                continue

            # Cycle through the columns in the channel.
            for xidx in range(self.cube.nxpix):
                if np.nanmax(channel[:, xidx]) < mph:
                    continue

                # Find the indices of the two largest peaks.
                yidx = detect_peaks(channel[:, xidx], mph=mph, mpd=mpd)
                if len(yidx) < 2:
                    continue
                pidx = channel[yidx, xidx].argsort()[::-1]
                yidx = yidx[pidx][:2]

                # Convert indices to polar coordinates.
                x = self.cube.xaxis[xidx]
                yf, yn = self.cube.yaxis[yidx]
                yc = 0.5 * (yf + yn)
                dy = max(yf - yc, yn - yc) / np.cos(inc)
                r = np.hypot(x - x0, dy)
                z = abs(yc - y0) / np.sin(inc)

                # Add coordinates to list. Apply some filtering.
                if np.isnan(r) or np.isnan(z):
                    continue
                if z > r / 2.:
                    continue
                if r > self.rout / self.dist:
                    continue
                if r < self.rin / self.dist:
                    continue

                # Include the brightness temperature.
                Tb = channel[yidx[0], xidx]

                # Include the coordinates to the list.
                coords += [[r * self.dist, z * self.dist, Tb]]
        return np.squeeze(coords).T

    def _weighted_percentiles(self, data, weights, pcnts=[.16, .50, .84]):
        '''Weighted percentiles.'''
        idx = np.argsort(data)
        sorted_data = np.take(data, idx)
        sorted_weights = np.take(weights, idx)
        cum_weights = np.add.accumulate(sorted_weights)
        scaled_weights = cum_weights - 0.5 * sorted_weights
        scaled_weights /= cum_weights[-1]
        spots = np.searchsorted(scaled_weights, pcnts)
        wp = []
        for s, p in zip(spots, pcnts):
            if s == 0:
                wp.append(sorted_data[s])
            elif s == data.size:
                wp.append(sorted_data[s-1])
            else:
                f1 = (scaled_weights[s] - p)
                f1 /= (scaled_weights[s] - scaled_weights[s-1])
                f2 = (p - scaled_weights[s-1])
                f2 /= (scaled_weights[s] - scaled_weights[s-1])
                wp.append(sorted_data[s-1] * f1 + sorted_data[s] * f2)
        return np.array(wp)
