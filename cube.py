"""
Class to aid in the calculation of the emission surface.
"""

from imgcube.imagecube import imagecube
from detect_peaks import detect_peaks
import numpy as np


class linecube:

    def __init__(self, path, rms=None, nsigma=3, dist=1.0, inc=0.0,
                 pa=False, downsample=False, **kwargs):
        """
        Read in the cube file using the `imagecube` class. Simple regridding
        functionality is provided to downsample and rotate.

        - Input Variables -

        path:       Relative path to the fits cube to use.
        rms:        The RMS noise in a line free channel in [K]. If not
                    provided, will attempt to estimate this using the end
                    `linefreechans`.
        nsigma:     All pixels below nsigma x rms will be clipped.
        dist:       Distance to the source in [pc], use to convert distances
                    from arcseconds to astronomical units.
        inc:        Inclination of the dist in [degrees]. Use as the default
                    when calculating the emission surface.
        pa:         Position angle of the disk in [degrees]. If provided will
                    rotate the image such that the minor axis is aligned with
                    the y axis.
        """

        # Read in the file and set defaults.
        self.path = path
        self.cube = imagecube(path)
        self.inc = inc
        self.dist = max(dist, 1.0)

        # Apply the rotation if necessary. First check if there's been a saved
        # rotated cube already as rotating takes a long time.
        if pa:
            try:
                newpath = path.replace('.fits', 'rotated%.1f.fits' % pa)
                self.cube = imagecube(newpath)
                self.path = newpath
            except:
                print("Rotating cube, this may take a while...")
                self.cube.rotate_cube(pa)

        # Downsample the data to boost the signal to noise in the channels.
        if downsample:
            self.cube.downsample_cube(downsample, center=True)

        # Use the noise to clip low signal pixels.
        self.data = self.cube.data
        if rms is not None:
            self.rms = rms
        else:
            self.rms = self.estimate_rms(kwargs.get('linefreechans', 5))
        self.clip = nsigma * self.rms
        self.data = np.where(abs(self.data) >= self.clip, self.data, 0.0)
        return

    def estimate_rms(self, N=5):
        """Estimate the noise from centre of the end N channels."""
        dx = int(self.cube.nxpix / 4)
        return np.nanstd([self.data[:N, dx:-dx, dx:-dx],
                          self.data[-N:, dx:-dx, dx:-dx]])

    def emission_surface(self, **kwargs):
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

    def bin_surface(self, coords, rmin=None, rmax=None, nbins=50):
        """
        Return the binned coordinates.

        - Input Variables -

        coords:     Coordinates from `find_surface`.
        rbins:      The centre of the bins used for the averaging in [au]. If
                    none are specified, will try to estimate them by sampling
                    between the minimum and maximum value with `nbins` points.
        nbins:      If `rbins` is not specified, the number of bins to use.

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
            bin = np.nanpercentile(coords[1][ridxs == r], [16, 50, 84])
            if np.isfinite(bin).all():
                zbins += [bin]
            else:
                zbins += [[np.nan, np.nan, np.nan]]
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
                if z > r:
                    continue

                # Include the brightness temperature.
                Tb = channel[yidx[0], xidx]

                # Include the coordinates to the list.
                coords += [[r * self.dist, z * self.dist, Tb]]
        return np.squeeze(coords).T
