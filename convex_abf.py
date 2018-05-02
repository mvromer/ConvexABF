# https://www.cs.ccu.edu.tw/~wtchu/courses/2012s_DSP/Lectures/Lecture%203%20Complex%20Exponential%20Signals.pdf
# http://www.labbookpages.co.uk/audio/beamforming/delayCalc.html
# https://skynet.ee.ic.ac.uk/papers/2011_C_MathsInDefence_VirtualLinearArray.pdf
# https://en.wikipedia.org/wiki/Sensor_array
# http://www.comm.utoronto.ca/~rsadve/Notes/ArrayTheory.pdf
# https://research.ijcaonline.org/volume61/number11/pxc3884758.pdf
# http://www.personal.psu.edu/faculty/m/x/mxm14/sonar/beamforming.pdf
# http://www.uio.no/studier/emner/matnat/ifi/INF5410/v12/undervisningsmateriale/foils/AdaptiveBeamforming.pdf

import functools
import itertools
import multiprocessing
import pathlib
import struct

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig

import picos as pic

from matplotlib import animation, rc
from IPython.display import HTML

#plt.ion()

class ElementTimeSeries:
    """
    Represents element time series data from an array comprising N elements sampled at R Hz each.

    The element time series data file contains a global header plus zero or more snapshots of data.
    Each snapshot is assumed to have S samples per element.

    Time is assumed to start at zero. Sample data is assumed to be floating point.

    The global header contains the following values in order:
        SamplesPerSnapshot (64-bit unsigned int) NOTE: This is a per channel per snapshot amount
        NumberElements     (64-bit unsigned int)
        SamplingRate       (64-bit floating point)
        SampleSizeBytes    (64-bit unsigned int)

    """
    MODE_READ = 0
    MODE_WRITE = 1

    @classmethod
    def create( cls, samplesPerSnapshot, numberElements, samplingRate, sampleSizeBytes,
               dataFileName ):
        """
        Creates an ETS object for writing.

        The new file will write snapshots of samples with dimensions samplesPerSnapshot-by-
        numberElements. Each sample will be sampleSizeBytes bytes large.

        """
        return cls( dataFileName, cls.MODE_WRITE, samplesPerSnapshot, numberElements,
                   samplingRate, sampleSizeBytes )

    @classmethod
    def open( cls, dataFileName ):
        """
        Creates an ETS object for reading.

        """
        return cls( dataFileName, cls.MODE_READ )

    def writeSnapshot( self, snapshot ):
        """
        Writes the given snapshot to the underlying ETS file.

        """
        numberSnapshotSamples, numberSnapshotElements = snapshot.shape
        assert numberSnapshotSamples == self.SamplesPerSnapshot, \
            "Number of samples per snapshot must be %d" % self.SamplesPerSnapshot
        assert numberSnapshotElements == self.NumberElements, \
            "Number of elements in snapshot must be %d" % self.NumberElements

        self._dataFile.write( snapshot.tobytes() )

    def readSnapshot( self ):
        """
        Reads the next snapshot from the underlying ETS file.

        """
        readSize = self.SamplesPerSnapshot * self.NumberElements
        snapshot = np.fromfile( self._dataFile, self._sampleDtype, readSize )
        if snapshot.size > 0:
            snapshot = np.reshape( snapshot, (self.SamplesPerSnapshot, self.NumberElements) )
        return snapshot

    def _readHeader( self ):
        self.SamplesPerSnapshot = self._readFieldFromFile( self.SamplesPerSnapshotHeaderFormat,
                                                          self.SamplesPerSnapshotHeaderSize )
        self.NumberElements = self._readFieldFromFile( self.NumberElementsHeaderFormat,
                                                      self.NumberElementsHeaderSize )
        self.SamplingRate = self._readFieldFromFile( self.SamplingRateHeaderFormat,
                                                   self.SamplingRateHeaderSize )
        self._sampleSizeBytes = self._readFieldFromFile( self.SampleSizeBytesHeaderFormat,
                                                        self.SampleSizeBytesHeaderSize )
        self._sampleDtype = self._makeSampleDtype( self._sampleSizeBytes )

    def _readFieldFromFile( self, fieldFormat, fieldSizeBytes ):
        # Note: struct.unpack returns a tuple even if there's only a single element.
        return struct.unpack( fieldFormat, self._dataFile.read( fieldSizeBytes ) )[0]

    def _writeHeader( self ):
        self._writeFieldToFile( self.SamplesPerSnapshot, self.SamplesPerSnapshotHeaderFormat )
        self._writeFieldToFile( self.NumberElements, self.NumberElementsHeaderFormat )
        self._writeFieldToFile( self.SamplingRate, self.SamplingRateHeaderFormat )
        self._writeFieldToFile( self._sampleSizeBytes, self.SampleSizeBytesHeaderFormat )

    def _writeFieldToFile( self, fieldValue, fieldFormat ):
        self._dataFile.write( struct.pack( fieldFormat, fieldValue ) )

    @staticmethod
    def _makeSampleDtype( sampleSizeBytes ):
        return np.dtype( "f%d" % sampleSizeBytes ) if sampleSizeBytes else None

    def __init__( self, dataFileName, mode,
                 samplesPerSnapshot=None, numberElements=None, samplingRate=None,
                 sampleSizeBytes=None ):
        assert mode in (self.MODE_READ, self.MODE_WRITE)
        assert dataFileName

        if mode == self.MODE_WRITE:
            assert samplesPerSnapshot > 0 and type(samplesPerSnapshot) is int, \
                "Samples per snapshot must be positive."
            assert numberElements > 0 and type(numberElements) is int, \
                "Number of elements must be positive integer."
            assert samplingRate > 0, "Sampling rate must be positive."
            assert sampleSizeBytes in (4, 8), "Sample size must be either 4 or 8 bytes."

        self._dataFileName = dataFileName
        self._dataFile = None
        self._mode = mode
        self.SamplesPerSnapshot = samplesPerSnapshot
        self.NumberElements = numberElements
        self.SamplingRate = samplingRate
        self._sampleSizeBytes = sampleSizeBytes
        self._sampleDtype = self._makeSampleDtype( self._sampleSizeBytes )

    def __enter__( self ):
        fileMode = "rb" if self._mode == self.MODE_READ else "wb"
        self._dataFile = open( self._dataFileName, fileMode )

        if self._mode == self.MODE_READ:
            self._readHeader()
        else:
            self._writeHeader()

        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self._dataFile.close()

    # File format constants
    NumberElementsHeaderSize = 8
    SamplingRateHeaderSize = 8
    SampleSizeBytesHeaderSize = 8
    SamplesPerSnapshotHeaderSize = 8

    NumberElementsHeaderFormat = "=Q"
    SamplingRateHeaderFormat = "=d"
    SampleSizeBytesHeaderFormat = "=Q"
    SamplesPerSnapshotHeaderFormat = "=Q"

class FourierSpectra:
    """
    Represents Fourier spectrum data for either a set of elements from an array or a set of beams
    formed by a beamformer, both generically referred to as channels.

    The Fourier spectrum data file contains a global header plus zero or more snapshots of spectral
    data. Each snapshot contains M frequency bins by N channels many complex128 values. If the value
    M is zero in the global header, then a full-band FFT is assumed, in which case the number of
    bins is the integer given by floor( FftLength / 2 ) + 1.

    The global header contains the following values in order:
        FftLength      (64-bit unsigned int)
        SamplingRate   (64-bit floating point)
        NumberChannels (64-bit unsigned int)
        NumberBins     (64-bit unsigned int)
        BinFrequencies (NumberBins many 64-bit floating point values)

    """
    MODE_READ = 0
    MODE_WRITE = 1

    @classmethod
    def createFullBand( cls, fftLength, samplingRate, numberChannels, dataFileName ):
        return cls( dataFileName, cls.MODE_WRITE, fftLength, samplingRate, numberChannels )

    @classmethod
    def create( cls, fftLength, samplingRate, numberChannels, numberBins, binFrequencies,
               dataFileName ):
        return cls( dataFileName, cls.MODE_WRITE, fftLength, samplingRate, numberChannels,
                   numberBins, binFrequencies )

    @classmethod
    def open( cls, dataFileName ):
        return cls( dataFileName, cls.MODE_READ )

    def writeSnapshot( self, snapshot ):
        numberSnapshotBins, numberSnapshotChannels = snapshot.shape
        assert numberSnapshotBins == self.NumberBins, \
            "Number of bins per snapshot must be %d" % self.NumberBins
        assert numberSnapshotChannels == self.NumberChannels, \
            "Number of channels in snapshot must be %d" % self.NumberChannels
        assert snapshot.dtype == self._binDtype, \
            "Snapshot data type must be %s" % self._binDtype

        self._dataFile.write( snapshot.tobytes() )

    def readSnapshot( self ):
        readSize = self.NumberBins * self.NumberChannels
        snapshot = np.fromfile( self._dataFile, self._binDtype, readSize )
        if snapshot.size > 0:
            snapshot = np.reshape( snapshot, (self.NumberBins, self.NumberChannels) )
        return snapshot

    def _readHeader( self ):
        self.FftLength = self._readFieldFromFile( self.FftLengthHeaderFormat,
                                                 self.FftLengthHeaderSize )
        self.SamplingRate = self._readFieldFromFile( self.SamplingRateHeaderFormat,
                                                   self.SamplingRateHeaderSize )
        self.NumberChannels = self._readFieldFromFile( self.NumberChannelsHeaderFormat,
                                                      self.NumberChannelsHeaderSize )
        self.NumberBins = self._readFieldFromFile( self.NumberBinsHeaderFormat,
                                                  self.NumberBinsHeaderSize )

        if self.NumberBins:
            self._isFullBand = False
            self.BinFrequencies = np.fromfile( self._dataFile, self.BinFrequencyDataType,
                                              self.NumberBins )
        else:
            self._isFullBand = True
            self.NumberBins = self._computeFullBandNumberBins( self.FftLength )
            self.BinFrequencies = self._computeFullBandBinFrequencies( self.SamplingRate,
                                                                      self.FftLength )

    def _readFieldFromFile( self, fieldFormat, fieldSizeBytes ):
        # Note: struct.unpack returns a tuple even if there's only a single element.
        return struct.unpack( fieldFormat, self._dataFile.read( fieldSizeBytes ) )[0]

    def _writeHeader( self ):
        self._writeFieldToFile( self.FftLength, self.FftLengthHeaderFormat )
        self._writeFieldToFile( self.SamplingRate, self.SamplingRateHeaderFormat )
        self._writeFieldToFile( self.NumberChannels, self.NumberChannelsHeaderFormat )
        self._writeFieldToFile( 0 if self._isFullBand else self.NumberBins,
                               self.NumberBinsHeaderFormat )

        if not self._isFullBand:
            self._dataFile.write( self.BinFrequencies.tobytes() )

    def _writeFieldToFile( self, fieldValue, fieldFormat ):
        self._dataFile.write( struct.pack( fieldFormat, fieldValue ) )

    def __init__( self, dataFileName, mode, fftLength=None, samplingRate=None, numberChannels=None,
                 numberBins=None, binFrequencies=None ):
        # We currently don't know if we're a full-band FFT or not.
        isFullBand = None

        # Other computed properties we won't know off the bat.
        binResolution = None

        assert mode in (self.MODE_READ, self.MODE_WRITE)
        assert dataFileName

        if mode == self.MODE_WRITE:
            assert fftLength > 0 and type(fftLength) is int, \
                "FFT length must be positive integer."
            assert samplingRate > 0, "Sampling rate must be positive."
            assert numberChannels > 0 and type(numberChannels) is int, \
                "Number of channels must be positive integer."

            binResolution = self._computeBinResolution( samplingRate, fftLength )

            if numberBins:
                assert numberBins > 0 and type(numberBins) is int, \
                    "Number of bins must be positive integer."

                maxNumberBins = self._computeFullBandNumberBins( fftLength )
                assert numberBins <= maxNumberBins, \
                    "Number of bins cannot exceed %d for FFT with length %d." % (maxNumberBins, fftLength)

                assert binFrequencies is not None, \
                    "Bin frequencies must be provided if number bins is given."
                assert binFrequencies.size == numberBins, \
                    "Number of bin frequencies must match number of bins."

                isFullBand = False
            else:
                assert binFrequencies is None, \
                    "Bin frequencies must NOT be provided if number of bins is not given."
                numberBins = self._computeFullBandNumberBins( fftLength )
                binFrequencies = self._computeFullBandBinFrequencies( samplingRate, fftLength )
                isFullBand = True

        self._dataFileName = dataFileName
        self._dataFile = None
        self._mode = mode
        self.FftLength = fftLength
        self.SamplingRate = samplingRate
        self.NumberChannels = numberChannels
        self.NumberBins = numberBins
        self.BinFrequencies = binFrequencies
        self.BinResolution = binResolution
        self._isFullBand = isFullBand
        self._binDtype = np.dtype( np.complex128 )

    @staticmethod
    def _computeBinResolution( samplingRate, fftLength ):
        return samplingRate / fftLength

    @staticmethod
    def _computeFullBandNumberBins( fftLength ):
        return int(np.floor( fftLength / 2 ) + 1)

    @staticmethod
    def _computeFullBandBinFrequencies( samplingRate, fftLength ):
        numberBins = FourierSpectra._computeFullBandNumberBins( fftLength )
        binResolution = FourierSpectra._computeBinResolution( samplingRate, fftLength )
        return np.arange( numberBins ) * binResolution

    def __enter__( self ):
        fileMode = "rb" if self._mode == self.MODE_READ else "wb"
        self._dataFile = open( self._dataFileName, fileMode )

        if self._mode == self.MODE_READ:
            self._readHeader()
        else:
            self._writeHeader()

        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self._dataFile.close()

    # File format constants
    BinFrequencyDataType = np.dtype( float )

    FftLengthHeaderSize = 8
    SamplingRateHeaderSize = 8
    NumberChannelsHeaderSize = 8
    NumberBinsHeaderSize = 8

    FftLengthHeaderFormat = "=Q"
    SamplingRateHeaderFormat = "=d"
    NumberChannelsHeaderFormat = "=Q"
    NumberBinsHeaderFormat = "=Q"

class ArrayGeometry:
    @classmethod
    def createUniformLinear( cls, numberElements, elementSpacing ):
        """
        Creates a geometry for a uniform linear array with the given number of elements and
        inter-element spacing given in meters.

        The array is oriented along the y axis with the center of the array coinciding with the
        origin of the local coordinate system. This allows an azimuthal angle of 0 degrees to
        coincide with broadside, which is customary in many sonar applications.

        """
        assert numberElements > 0 and type(numberElements) is int, \
            "Number of elements must be a positive integer."
        assert elementSpacing > 0, "Element spacing must be positive."

        arrayLength = (numberElements - 1) * elementSpacing
        midpoint = arrayLength / 2.0
        posX = np.zeros( numberElements )
        posY = np.linspace( 0.0, arrayLength, numberElements, endpoint=True ) - midpoint
        posZ = np.zeros( numberElements )
        return cls( posX, posY, posZ )

    def __init__( self, posX, posY, posZ ):
        assert posX.shape == posY.shape and posY.shape == posZ.shape, \
            "Geometry coordinate arrays must have same shape."

        self.NumberElements = posX.size
        self.X = posX
        self.Y = posY
        self.Z = posZ

class Hydrophone:
    """
    Defines the characteristics of a hydrophone element within an array.

    The primary attribute is the hydrophone's sensitivity, given in dB (re 1 V/uPa). This can be
    used to determine the amount of voltage produced by the hydrophone when subjected to an acoustic
    wave with a specific average sound pressure level.

    """
    def __init__( self, sensitivity=-180.0 ):
        self.Sensitivity = sensitivity

        # Hydrophone sensitivity in dB is given by S = 20 * log10( (X V/uPa) / (1 V/uPa) ), where X
        # is the number of volts produced by the hydrophone per micropascal of sound pressure.
        #
        # http://resource.npl.co.uk/acoustics/techguides/concepts/sen.html
        # https://electronics.stackexchange.com/questions/96205/how-to-convert-volts-in-db-spl
        # http://www.indiana.edu/~emusic/etext/acoustics/chapter1_amplitude4.shtml
        self._voltsPerMicropascal = 10.0 ** (self.Sensitivity / 20.0)

    def computeOutputVoltage( self, incomingSpl ):
        """
        Calculates the voltage produced by the hydrophone when it is subjected to an acoustic wave
        with the given average sound pressure level in dB SPL (re 1 uPa).

        """
        # Sound pressure level in dB (re 1 uPa) is given by SPL = 20 * log10( P / 1 uPa ), where P
        # is the pressure of the acoustic wave in micropascals.
        #
        # http://www.arc.id.au/SoundLevels.html
        # https://fas.org/man/dod-101/sys/ship/acoustics.htm
        # https://oceanexplorer.noaa.gov/explorations/sound01/background/acoustics/acoustics.html
        return 10.0 ** (incomingSpl / 20.0) * self._voltsPerMicropascal

class ArraySimulator:
    def __init__( self, geometry, hydrophone, samplingRate, snapshotDuration, speedOfSound=1500.0 ):
        """
        Sets up a new array simulator for an array with the given geometry and that consists of
        identical elements defined by the given hydrophone type.

        The array samples acoustic signals at the given sampling rate and produces snapshots of
        samples lasting the given duration in seconds. Signals propagate at the given sound speed in
        meters per second.

        """
        samplesPerSnapshot = samplingRate * snapshotDuration
        assert samplesPerSnapshot > 0 and  np.floor( samplesPerSnapshot ) == samplesPerSnapshot, \
            "Product of sampling rate and snapshot duration must be a positive integer."

        self._samplesPerSnapshot = int(samplesPerSnapshot)
        self._speedOfSound = speedOfSound
        self._snapshotDuration = snapshotDuration
        self._samplingRate = samplingRate
        self._hydrophone = hydrophone
        self._geometry = geometry
        self._noiseGenerator = None
        self._targets = []
        self._targetTimeDelays = []

    def setNoiseGenerator( self, noiseGenerator ):
        self._noiseGenerator = noiseGenerator

    def addTarget( self, target ):
        self._targets.append( target )
        self._targetTimeDelays.append( self.computeElementTimeDelay( target.Position ) )

    def computeElementTimeDelay( self, sourcePosition ):
        """
        Computes the time delays for each element in the array when a signal propagating from a
        source located at the given PositionAzEl impinges on the phase center of the array, which is
        taken to be the origin of the local coordinate system.

        If the range of the signal origin is finite, then the source is presumed to be in the near
        field. If the range goes out to infinity, then the source is presumed to be in the far
        field. Signals for sources in the far field appear to have planar wavefronts to the array
        elements.

        Note that only sources in the far field are supported at this time.

        """
        if sourcePosition.Range < np.inf:
            raise NotImplementedError( "Near field sources not supported at this time." )

        # Derived from Optimum Array Procesing page 29. If A is a unit vector defining the direction
        # of propagation for the planar wavefront impinging on the array's phase center and P is the
        # position of an array element, then the time delay tau for the element is given by
        # (A . P) / c, where c is the speed of the wavefront's propagation. A is defined in terms of
        # the spherical coordinates of the signal's source as follows:
        #
        #     [ -sin(theta) * cos(phi)
        #       -sin(theta) * sin(phi)
        #       -cos(theta) ]
        #
        xA, yA, zA = sphericalToUnitCartesian( sourcePosition.Azimuth, sourcePosition.PolarAngle )
        dotProducts = np.sum( np.stack( (xA * self._geometry.X,
                                         yA * self._geometry.Y,
                                         zA * self._geometry.Z) ), axis=0 )
        return -1.0 / self._speedOfSound * dotProducts

    def simulate( self, numberSnapshots, fileName ):
        """
        Generates the noise and target signals in the simulated array environment for the given
        number of snapshots and writes the time series for each snapshot to the specified file as an
        element time series file.

        """
        # Use the default Numpy data type for each of our samples, which should be an 8-byte float,
        # or equivalently a C double.
        sampleByteSize = np.dtype( None ).itemsize

        with ElementTimeSeries.create( self._samplesPerSnapshot, self._geometry.NumberElements,
                                      self._samplingRate, sampleByteSize, fileName ) as etsFile:
            snapshotShape = (self._samplesPerSnapshot, self._geometry.NumberElements)

            simDuration = numberSnapshots * self._snapshotDuration
            snapshotStartTimes = np.linspace( 0.0, simDuration, numberSnapshots, endpoint=False )
            for snapshotStartTime in snapshotStartTimes:
                # Initialize our snapshot based on whether we have a noise generator or not.
                if self._noiseGenerator:
                    snapshot = self._noiseGenerator.generateSamples( snapshotShape )
                else:
                    snapshot = np.zeros( snapshotShape )

                # Setup the time range over which we wll be generating samples for each target.
                snapshotEndTime = snapshotStartTime + self._snapshotDuration
                snapshotTime = np.linspace( snapshotStartTime, snapshotEndTime,
                                           self._samplesPerSnapshot, endpoint=False )

                # Generate signals for each contact and mix them into our snapshot.
                for (target, targetTimeDelay) in zip( self._targets, self._targetTimeDelays ):
                    for iElement in range( self._geometry.NumberElements ):
                        samples, iTargetTime = target.generateSamples( snapshotTime,
                                                                      self._hydrophone,
                                                                      targetTimeDelay[iElement] )
                        snapshot[iTargetTime, iElement] += samples

                # Finalize the snapshot and write it out.
                etsFile.writeSnapshot( snapshot )

# http://www.ele.uri.edu/courses/ele447/Slides/CurMir/Chapter_07.pdf
# https://electronics.stackexchange.com/questions/37568/units-of-noise-spectral-density
# http://www.scholarpedia.org/article/Signal-to-noise_ratio
class WhiteNoiseGenerator:
    @classmethod
    def createFromNoiseDensity( cls, noiseDensity, bandwidth ):
        """
        Generates a new white noise generator given the noise density and the bandwidth over which
        the noise is present.

        White noise has equal power across all frequencies. This means that the total noise power
        is give by the product of the noise density and the bandwidth. For a real-valued signal
        (which in this case all of our acoustic signals are real-valued), we have a two-sided
        spectrum consisting of both positive and negative frequencies, so the noise power is
        spread over both halves. This means that the total noise power is half the product of the
        noise density and the bandwidth.

        The noise power is the variance of the noise itself, so we can get the standard deviation
        of our Gaussian white noise process by taking the square root of the computed noise power.
        Another way to think of this is that the power of the noise is proportional to the squared
        voltage of the noise, so taking the square root gives us the signal's voltage.

        """
        noisePower = 0.5 * noiseDensity * bandwidth
        return cls( np.sqrt( noisePower ) )

    @classmethod
    def createFromSoundPressureLevel( cls, soundPressureLevel, hydrophone ):
        """
        Generates a new white noise generator given the average sound pressure level in dB (re 1
        uPa) of the noise and the hydrophone subjected to the noise.

        """
        return cls( hydrophone.computeOutputVoltage( soundPressureLevel ) )

    def __init__( self, standardDeviation ):
        self._noiseStandardDeviation = standardDeviation

    def setSeed( self, seed ):
        np.random.seed( seed )

    def generateSamples( self, shape ):
        return np.random.normal( scale=self._noiseStandardDeviation, size=shape )

class PositionAzEl:
    """
    Defines a position in 3D space by the triplet (range, elevation, azimuth).

    We presume a 3D coordinate system where the xy plane represents the ground plane, and the z axis
    is the vertical axis. For a point P in 3D space, the spherical coordinates are (r, theta, phi).

    The range or radial distance in meters from the origin to P is r.

    The azimuthal angle measured in degrees from the positive x axis to the orthogonal projection of
    P on the xy plane is phi.

    The polar angle measured in degrees from the positive z axis to P is theta. The elevation angle
    in degrees is 90 - theta.

    """
    def __init__( self, azimuth, elevation=0.0, range=np.inf ):
        assert range > 0, "Range must be positive."
        self.Range = range
        self.Azimuth = azimuth
        self.Elevation = elevation

    @property
    def PolarAngle( self ):
        return elevationToPolarAngle( self.Elevation )

class Target:
    """
    Represents a target at a given PositionAzEl in 3D space emitting a signal at a given linear
    frequency in Hertz and average sound pressure level in dB (re 1 uPa).

    The time window during which the target is emitting its signal can be restricted by specifying a
    start and/or stop time.

    """
    def __init__( self, position, frequency, soundPressureLevel, signalGenerator,
                 startTime=-np.inf, stopTime=np.inf ):
        self.Position = position
        self._frequency = frequency
        self._soundPressureLevel = soundPressureLevel
        self._signalGenerator = signalGenerator
        self._startTime = startTime
        self._stopTime = stopTime

    def generateSamples( self, time, hydrophone, timeDelay=0.0 ):
        """
        Generates a time series representing the voltage over the given time range that is produced
        by the given hydrophone when it is subjected to acoustic signal emitted by this target.

        This returns both the voltage samples and an index array corresponding to those times in the
        input time range samples were generated for. Returns None if the requested time is outside
        this target's active time window.

        """
        if self._stopTime < time[0] or self._startTime > time[-1]:
            return (None, None)

        # Subselect the time indices that lie within this target's active time window. Bitwise AND
        # trick for subselecting values of an ndarray: https://stackoverflow.com/a/3030662/562685
        iTargetTime = np.nonzero( (self._startTime <= time) & (time <= self._stopTime) )
        rmsVolts = hydrophone.computeOutputVoltage( self._soundPressureLevel )
        samples = self._signalGenerator.generateSamples( time[iTargetTime] - timeDelay,
                                                        self._frequency, rmsVolts )
        return (samples, iTargetTime)

class SineGenerator:
    def generateSamples( self, time, frequency, rmsAmplitude ):
        # Relationship between RMS and peak amplitude for sine wave: rms = peak / sqrt(2)
        # https://en.wikipedia.org/wiki/Root_mean_square#In_waveform_combinations
        peakAmplitude = np.sqrt( 2 ) * rmsAmplitude
        return peakAmplitude * np.sin( 2.0 * np.pi * frequency * time )

class SawtoothGenerator:
    def __init__( self, rising=True ):
        self._rising = rising

    def generateSamples( self, time, frequency, rmsAmplitude ):
        # Relationship between RMS and peak amplitude for sawtooth wave: rms = peak / sqrt(3)
        # https://en.wikipedia.org/wiki/Root_mean_square#In_waveform_combinations
        peakAmplitude = np.sqrt( 3 ) * rmsAmplitude
        width = 1.0 if self._rising else 0.0
        # NOTE: In testing, it was observed that certain time values can result in sig.sawtooth
        # producing a NaN. These ought to be filtered out in subsequent processing to avoid bad
        # stuff from happening in our calculations.
        return peakAmplitude * sig.sawtooth( 2.0 * np.pi * frequency * time, width )

class TriangleGenerator:
    def generateSamples( self, time, frequency, rmsAmplitude ):
        # Relationship between RMS and peak amplitude for triangle wave: rms = peak / sqrt(3)
        # https://en.wikipedia.org/wiki/Root_mean_square#In_waveform_combinations
        peakAmplitude = np.sqrt( 3 ) * rmsAmplitude
        return peakAmplitude * sig.sawtooth( 2.0 * np.pi * frequency * time, width=0.5 )

class RectangularWindow:
    def get( self, numberPoints ):
        return sig.boxcar( numberPoints )

class HannWindow:
    def get( self, numberPoints ):
        return sig.hann( numberPoints, sym=False )

class HammingWindow:
    def get( self, numberPoints ):
        return sig.hamming( numberPoints, sym=False )

class TukeyWindow:
    def __init__( self, alpha ):
        assert 0 <= alpha <= 1, "Tukey window control parameter must be in [0, 1]."
        self._alpha = alpha

    def get( self, numberPoints ):
        return sig.tukey( numberPoints, self._alpha, sym=False )

class BlackmanWindow:
    def get( self, numberPoints ):
        return sig.blackman( numberPoints, sym=False )

class FourierTransformer:
    def __init__( self, overlap=0.5, window=RectangularWindow() ):
        assert 0 <= overlap < 1, "Overlap must be in [0, 1)"
        assert window, "A window must be provided."
        self._overlap = overlap
        self._window = window

    def transformTimeSeries( self, etsFileName, fftFileName ):
        with ElementTimeSeries.open( etsFileName ) as inputEts:
            fftLength = inputEts.SamplesPerSnapshot
            numberChannels = inputEts.NumberElements
            samplingRate = inputEts.SamplingRate

            with FourierSpectra.createFullBand( fftLength, samplingRate, numberChannels,
                                               fftFileName ) as outputFft:
                # We maintain three FFT buffers during the transform. The first one is the buffer we
                # will actually window and transform. The second and third ones together will
                # contain the two most recently read snapshots of element time series data.
                #
                # Let iOverlap = floor( fftLength * overlap ). let A and B contain our two most
                # recently read ETS snapshots and T be our transform buffer. Let s1 be the slice
                # [0:iOverlap] and s2 be the slice [iOverlap:]. Let T1 = T[s1] and T2 = [s2].
                #
                # To implement our overlap, we setup two sequences of tuples:
                #
                #     D = ( (A,A), (A,B), (B,B), (B,A) )
                #     S = ( (s1,s2), (s2,s1) )
                #
                # Each transform iteration, cycle to the next element in D and S. The tuple from D
                # will tell us which input snapshots we will read from to populate T1 and T2,
                # respectively. The tuple from S will tell us which slices to pull from each of our
                # input snapshots to ensure the right samples are populated in T1 and T2,
                # respectively. When either D or S have exhausted all of their tuples, they cycle
                # back to the first tuple in their respective sequence.
                #
                # As an example, the following illustrates how T1 and T2 are populated in the first
                # 4 iterations of the transform loop.
                #
                #     Iteration 1: T1 = A[s1], T2 = A[s2]
                #     Iteration 2: T1 = A[s2], T2 = B[s1]
                #     Iteration 3: T1 = B[s1], T2 = B[s2]
                #     Iteration 4: T1 = B[s2], T2 = A[s1]
                #
                # A sequence C can be similarly built up containing a list of callbacks to call at
                # the start of each transform iteration. These callbacks can signal to the
                # transformer when to break out of the transformation loop (due to all input data
                # being transformed) and can perform zero padding (when the snapshot used to
                # populate T2 is empty).
                #
                numberBins = outputFft.NumberBins
                inputSnapshot1 = self._readAndFilterSnapshot( inputEts )
                inputSnapshot2 = np.empty( inputSnapshot1.shape )
                transformSamples = np.empty( inputSnapshot1.shape )

                if inputSnapshot1.size == 0:
                    return

                # Build up our collection of cycles we will iterate through with each transform.
                iOverlap = fftLength - int( fftLength * self._overlap )
                slice1 = (np.s_[0:iOverlap], np.s_[:])
                slice2 = (np.s_[iOverlap:], np.s_[:])

                snapshotCycle = itertools.cycle( ((inputSnapshot1, inputSnapshot1),
                                                  (inputSnapshot1, inputSnapshot2),
                                                  (inputSnapshot2, inputSnapshot2),
                                                  (inputSnapshot2, inputSnapshot1)) )

                sliceCycle = itertools.cycle( ((slice1, slice2),
                                               (slice2, slice1)) )
                callbackCycle = itertools.cycle( (None, self._readAndCheckNextSnapshot) )

                # Compute the window we will apply to the transform buffer.
                window = self._window.get( fftLength )

                while True:
                    # Cycle our input buffers and input slices.
                    inputSnapshot1, inputSnapshot2 = next( snapshotCycle )
                    inputSnapshotSlice1, inputSnapshotSlice2 = next( sliceCycle )
                    callback = next( callbackCycle )

                    # If it is time to do so, call our callback responsible for reading in the next
                    # snapshot and applying any zero padding if necessary.
                    terminate = False
                    if callback:
                        terminate = callback( inputEts, inputSnapshot2, inputSnapshotSlice2 )

                    # Build the transform buffer, window it, and FFT it.
                    transformSamples[inputSnapshotSlice1] = inputSnapshot1[inputSnapshotSlice1]
                    transformSamples[inputSnapshotSlice2] = inputSnapshot2[inputSnapshotSlice2]

                    np.apply_along_axis( lambda samples : samples * window,
                                        axis=0, arr=transformSamples )
                    transformSpectra = sp.fft( transformSamples, axis=0 )

                    # Write out only those bins that correspond to DC and our positive frequencies.
                    # The negative frequencies are just complex conjugates of the real ones, and
                    # thus are redundant.
                    outputFft.writeSnapshot( transformSpectra[0:numberBins, :] )

                    # Terminate if our callback signaled that we should terminate.
                    if terminate:
                        break

    @staticmethod
    def _readAndFilterSnapshot( etsFile ):
        newSnapshot = etsFile.readSnapshot()
        if newSnapshot.size > 0:
            # Our array simulator can sometimes produce NaNs if the input time values are denormal.
            # We need to filter these out of the snapshot so they don't balls up the Fourier
            # transform. It's just like dealing with bad samples from real arrays...
            newSnapshot[np.isnan( newSnapshot )] = 0.0
        return newSnapshot

    @staticmethod
    def _readAndCheckNextSnapshot( etsFile, snapshot, snapshotSlice ):
        """
        Reads the next snapshot from the given ETS file and checks to see if it empty. If it is not,
        then the given snapshot buffer is populated with the new snapshot data, and False is
        returned signifying that the transformation loop should NOT terminate.

        If there is no new snapshot data, then the given snapshot buffer will be zero filled for the
        given slice of elements. True will be returned signifying that the transformation loops
        should terminate.

        """
        newSnapshot = FourierTransformer._readAndFilterSnapshot( etsFile )
        if newSnapshot.size == 0:
            snapshot[snapshotSlice] = 0.0
            return True
        else:
            snapshot[:] = newSnapshot
            return False

class Beam:
    def __init__( self, azimuth, elevation=0.0 ):
        self.Azimuth = azimuth
        self.Elevation = elevation

    @property
    def PolarAngle( self ):
        return elevationToPolarAngle( self.Elevation )

class FrequencyBand:
    def __init__( self, startHertz = -np.inf, stopHertz = np.inf ):
        assert startHertz == -np.inf or startHertz >= 0.0, \
            "Band start must be -INF or non-negative finite."
        assert stopHertz > 0.0, "Band stop must be positive."
        assert stopHertz > startHertz, "Band stop must be greater than band start."
        self.StartHertz = startHertz
        self.StopHertz = stopHertz

def elevationToPolarAngle( elevation ):
    return 90.0 - elevation

class Beamformer:
    def __init__( self, arrayGeometry, outputBeams, bandToProcess, snapshotAverageCount,
                 speedOfSound, memoryBudgetMB = 1024 ):
        assert snapshotAverageCount > 0

        self._arrayGeometry = arrayGeometry
        self._outputBeams = outputBeams
        self._bandToProcess = bandToProcess
        self._snapshotAverageCount = snapshotAverageCount
        self._speedOfSound = speedOfSound
        self._memoryBudgetMB = memoryBudgetMB

    @property
    def _NumberElements( self ):
        return self._arrayGeometry.NumberElements

    @property
    def _NumberBeams( self ):
        return len( self._outputBeams )

    def process( self, inputFileName, outputFileName, weightAlgorithm ):
        with FourierSpectra.open( inputFileName ) as inputFft:
            # Figure out the frequency band we're going to process.
            iStartBin, iStopBin = self._getBinIndicesToProcess( inputFft.BinFrequencies )
            numberBins = iStopBin - iStartBin
            binFrequencies = inputFft.BinFrequencies[iStartBin:iStopBin]
            print( "Beamforming frequency band from %g Hz to %g Hz." %
                  (binFrequencies[0], binFrequencies[-1]) )
            print( "Averaging %d snapshots per CSM." % self._snapshotAverageCount )

            with FourierSpectra.create( inputFft.FftLength, inputFft.SamplingRate,
                                       self._NumberBeams, numberBins, binFrequencies,
                                       outputFileName ) as outputFft:
                self._innerProcess( inputFft, outputFft, iStartBin, iStopBin, weightAlgorithm )

    def _innerProcess( self, inputFft, outputFft, iStartBin, iStopBin, weightAlgorithm ):
        numberBufferedSnapshots = self._snapshotAverageCount
        numberInputBins = inputFft.NumberBins
        numberOutputBins = iStopBin - iStartBin
        numberBeams = self._NumberBeams
        numberElements = self._NumberElements

        # First precompute all of our steering vectors that we will use.
        steeringVectors = self._computeSteeringVectors( inputFft.BinFrequencies[iStartBin:iStopBin] )

        # Allocate the output snapshot.
        outputSnapshot = np.empty( shape=(numberOutputBins, numberBeams), dtype=np.complex )

        # Allocate the input snapshot buffer.
        inputSnapshots = np.empty( shape=(numberBufferedSnapshots, numberInputBins, numberElements),
                                  dtype=np.complex )

        # Allocate the process pool now so we aren't constantly setting it up and tearing it down
        # later in this method.
        numberProcs = multiprocessing.cpu_count()
        with multiprocessing.Pool( numberProcs ) as pool:
            # Keep processing until we have no more CSMs available for processing.
            snapshotsToKeep = 0
            iOutputSnapshot = 0
            while True:
                numberNewSnapshots = self._readNextSnapshots( inputFft, inputSnapshots, snapshotsToKeep )
                numberValidSnapshots = snapshotsToKeep + numberNewSnapshots
                numberAvailableCsms = numberValidSnapshots - self._snapshotAverageCount + 1
                if numberAvailableCsms <= 0:
                    break

                # We buffer up enough snapshots to compute exactly 1 CSM.
                assert numberAvailableCsms == 1
                iCurrentCsm = 0
                iCsmSnapshotSliceStart = iCurrentCsm
                iCsmSnapshotSliceStop = iCsmSnapshotSliceStart + self._snapshotAverageCount
                iCsmSnapshot = iCsmSnapshotSliceStop - 1
                csmSnapshotSlice = slice( iCsmSnapshotSliceStart, iCsmSnapshotSliceStop )

                # Build up the argument lists that represent the work necessary to compute the
                # weights for each CSM we process.
                print( "Computing weights for output snapshot %d" % (iOutputSnapshot+1) )
                processArgs = ((inputFft.BinFrequencies[iInputBin],
                                inputSnapshots[csmSnapshotSlice, iInputBin, :],   # csmSnapshots
                                steeringVectors[(iInputBin -  iStartBin), :, :],  # steeringVectors
                                weightAlgorithm)                                  # weightAlgorithm
                               for iInputBin in range( iStartBin, iStopBin ))

                weights = pool.starmap( self._processCsm, processArgs, chunksize=10 )

                # Apply the weights for each CSM processed.
                for iInputBin in range( iStartBin, iStopBin ):
                    iOutputBin = iInputBin - iStartBin
                    currentCsmSnapshot = inputSnapshots[iCsmSnapshot, iInputBin, :]
                    # Beamformer output for a single (frequency, beam) pair is given by w*x, where w
                    # is the Nx1 vector of weights computed for that (frequency, beam), and x is the
                    # Nx1 vector containing the snapshot spectral data for that frequency's bin.
                    #
                    # https://www.acoustics.asn.au/conference_proceedings/AAS2005/papers/8.pdf
                    #
                    outputSnapshot[iOutputBin, :] = np.apply_along_axis( \
                                  lambda binWeights: np.dot( binWeights.conj(), currentCsmSnapshot ),
                                  axis=1, arr=weights[iOutputBin] )

                    # Alternate implementation using einsum. This may be faster, but we use the
                    # above to match the output of the _precomputeProcess that we had in a previous
                    # commit.
                    #
                    #outputSnapshot[iOutputBin, :] = np.einsum( "ij,j->i",
                    #                                          currentWeights.conj(),
                    #                                          currentCsmSnapshot )


                # Write out this snapshot and compute how many snapshots we will keep for the next
                # iteration. Note that number should be one less than the number we buffer.
                outputFft.writeSnapshot( outputSnapshot )
                iOutputSnapshot += 1
                snapshotsToKeep = numberValidSnapshots - numberAvailableCsms
                assert snapshotsToKeep == (numberBufferedSnapshots - 1)

    @staticmethod
    def _processCsm( binFrequency, csmSnapshots, steeringVectors, weightAlgorithm ):
        # Form the cross-spectral matrix from the given snapshots.
        #
        # To form the CSM for the current frequency, we compute the XX*, where X is an N-by-J matrix
        # containing the next J snapshots of spectral data for all N channel where {.}* is the
        # conjugate transpose.
        #
        # Note that our input FFT data has dimensionality of (snapshot, frequency, channel), and
        # when we take the slice of our snapshot buffer corresponding to the matrix X, we end up
        # with a J-by-N view of the snapshot data used to compute the current CSM. Our view needs to
        # be transposed (but not conjugated) before multiplying, which is why it looks like we're
        # computing X'conj(X) instead (where {.}' is the non-conjugate transpose).
        #
        #print( "Processing bin frequency %g Hz" % binFrequency )
        numberSnapshots = csmSnapshots.shape[0]
        csm = (1.0 / numberSnapshots) * (csmSnapshots.T @ csmSnapshots.conj())

        # Scan over all beams and compute a set of weights for each (frequency, beam) combination.
        numberBeams, numberElements = steeringVectors.shape
        outputWeights = np.empty( shape=(numberBeams, numberElements), dtype=np.complex )
        for iBeam in range( numberBeams ):
            currentSteeringVector = steeringVectors[iBeam, :]
            outputWeights[iBeam, :] = weightAlgorithm.compute( csm, currentSteeringVector )

        return outputWeights

    def _getBinIndicesToProcess( self, binFrequencies ):
        return getBinIndicesForBand( self._bandToProcess, binFrequencies )

    def _readNextSnapshots( self, inputFft, snapshotBuffer, snapshotsToKeep ):
        numberBufferedSnapshots = snapshotBuffer.shape[0]
        snapshotsToRead = numberBufferedSnapshots - snapshotsToKeep

        if snapshotsToRead == 0 :
            return 0

        # Pack old snapshots that we're keeping to the front of the snapshot buffer. If we're not
        # keeping any snapshots, then there's no need to shuffle since we're going to overwrite the
        # entire snapshot buffer.
        if snapshotsToKeep > 0:
            snapshotBuffer[0:snapshotsToKeep, :] = snapshotBuffer[snapshotsToRead:, :]

        numberNewSnapshots = 0
        for iNewSnapshot in range( snapshotsToKeep, numberBufferedSnapshots ):
            newSnapshot = inputFft.readSnapshot()
            if newSnapshot.size == 0:
                break
            snapshotBuffer[iNewSnapshot, :] = newSnapshot
            numberNewSnapshots += 1

        return numberNewSnapshots

    def _computeSteeringVectors( self, binFrequencies ):
        """
        Computes a steering vector for each input frequency for every output beam this beamformer
        is configured for.

        The resulting 3D array of steering vectors is M x Q x N, where M is the number of
        frequencies, Q is the number of beams, and N is the number of array elements. The way to
        view this is that the mth slice is a set of Q steering vectors that steer the N elements of
        the array in the directions of the corresponding beams.

        Note that the steering vectors can consume A LOT of memory. Typically the frequency
        dimension is the most dominant one. As such, one should be careful when picking how many
        frequencies worth of steering vectors to compute.

        """
        # From Optimum Array Processing (p. 30), the steering vector (or array manifold vector) is
        # a function of a directional wavenumber, i.e., a function of frequency and look direction
        # (which in turn is defined by an azimuth phi and a polar angle theta). The steering vector
        # v(k) = [exp(-j * (k . p1))  |  exp(-j * (k . p2))  | ...  |  exp(-j * (k . pN))].
        #
        # The Cartesian coordinate for each array element I is given by pI. The wavenumber k is
        # given by k = -2 * pi * f / c * A, where f is the frequency of the plane wave corresponding
        # to the computed steering vector and c is the speed at which the plane wave is propagating.
        # A is a unit vector defined in terms of the spherical coordinates of the plane wave's
        # source and is given by the following:
        #
        #     [ -sin(theta) * cos(phi)
        #       -sin(theta) * sin(phi)
        #       -cos(theta) ]
        #
        # The imaginary unit is given by j.
        #
        numberElements = self._NumberElements
        numberBeams = self._NumberBeams
        twoPiOverC = 2.0 * np.pi / self._speedOfSound

        beamCartesian = (sphericalToUnitCartesian( beam.Azimuth, beam.PolarAngle ) \
                         for beam in self._outputBeams)
        beamCartesianFlat = itertools.chain.from_iterable( beamCartesian )

        elementCartesian = zip( self._arrayGeometry.X,
                               self._arrayGeometry.Y,
                               self._arrayGeometry.Z )
        elementCartesianFlat = itertools.chain.from_iterable( elementCartesian )

        # NOTE: np.fromiter only works on 1D iterables to make 1D arrays, so we flatten our beam
        # and element Cartesian coordinates to create the array and then reshape them respectively
        # to the appropriate Q x 3 and N x 3 arrays.
        return np.exp( 1j *
                      np.einsum( 'm,qi,ni->mqn',
                                twoPiOverC * binFrequencies,
                                np.fromiter( beamCartesianFlat, np.float, numberBeams * 3 ).reshape( (numberBeams, 3) ),
                                np.fromiter( elementCartesianFlat, np.float, numberElements * 3 ).reshape( (numberElements, 3) ) ) )

def sphericalToUnitCartesian( azimuth, polar ):
    polar = np.radians( polar )
    azimuth = np.radians( azimuth )
    sinPolar = np.sin( polar )
    sinAzimuth = np.sin( azimuth )
    cosPolar = np.cos( polar )
    cosAzimuth = np.cos( azimuth )

    return np.array( (sinPolar * cosAzimuth,
                      sinPolar * sinAzimuth,
                      cosPolar) )

def getBinIndicesForBand( frequencyBand, binFrequencies ):
    startHertz = frequencyBand.StartHertz
    stopHertz = frequencyBand.StopHertz
    if startHertz == -np.inf:
        iStartBin = 0
    else:
        iStartBin = -1
        while (iStartBin + 1) < binFrequencies.size and binFrequencies[iStartBin + 1] <= startHertz:
            iStartBin += 1

    if stopHertz == np.inf:
        iStopBin = binFrequencies.size
    else:
        iStopBin = binFrequencies.size
        while iStopBin > 0 and binFrequencies[iStopBin - 1] > stopHertz:
            iStopBin -= 1

    return (iStartBin, iStopBin)

class ConventionalWeights:
    """
    Computes the conventional beamformer weights.

    A conventional beamformer is simply a delay-and-sum beamformer in the time domain. In the
    frequency domain for a narrowband signal, the time delay is approximated by a phase shift. The
    conventional beamforming weights are equal to the steering vector corresponding with the desired
    direction of arrival.

    """
    def compute( self, csm, steeringVector ):
        return np.copy( steeringVector )

class CaponWeights:
    """
    Computes the standard Capon beamformer weights.

    The Capon beamformer is also known as the minimum variance distortionless response (MVDR)
    beamformer. It is the set of weights that minimize the power of the sample cross spectral matrix
    while maintaining unity gain in the when focusing in on a particular direction of arrival (as
    given by the steering vector).

    """
    def compute( self, csm, steeringVector ):
        # Reshape the steering vector so that it's viewed as a 2D array. This just keeps things
        # consistent in our model, where our weight vector is actually shaped like a 2D array.
        # We'll be sure to reshape the final weights as a 1D array after the optimization runs.
        numberElements = steeringVector.size
        steeringVector2D = steeringVector.reshape( (numberElements, 1) )

        # Picos wasn't able to properly handle the quadratic objective function w*Rw, where R our
        # CSM and w is our weight vector. It appears that it can't verify that the objective is
        # real-valued since it's a quadratic expression. To get around this, we hoist the problem
        # from a set of complex-valued matrices and vectors into real-valued ones.
        #
        # For the CSM, we let the lifted version be [Re(csm) -Im(csm) ; Im(csm) Re(csm)]. For the
        # steering vector, we formulate it as [Re(steeringVector) ; Im(steeringVector)].
        #
        liftedCsm = np.bmat( [[np.real( csm ), -np.imag( csm )],
                               [np.imag( csm ), np.real( csm )]] )

        liftedSteeringVector = np.vstack( [np.vstack( [np.real( sv ),
                                                       np.imag( sv )] ) for sv in steeringVector2D] )

        P = pic.Problem()
        weights = P.add_variable( "weights", (2 * numberElements, 1) )
        csmParam = pic.new_param( "csm", liftedCsm )
        steeringVectorParam = pic.new_param( "steeringVector", liftedSteeringVector )

        # Our distortionless response contraint is w*v == 1.0, where w is the weight vector and v is
        # the steering vector. Since the constraint uses the complex conjugate, we formulate the
        # the lifted weight vector (for this constraint) as [Re(weights) ; -Im(weights)]. Since we
        # want to constrain the left hand side to be unity, i.e., 1 + 0j, the right hand side turns
        # from the scalar 1 to the unit vector [1 ; 0].
        w_r = weights[0:numberElements]
        w_i = weights[numberElements:]
        top = functools.reduce( lambda x, y: x & y,
                               (wi_r & wi_i for wi_r, wi_i in zip( w_r, w_i )) )
        bottom = functools.reduce( lambda x, y: x & y,
                                  (-wi_i & wi_r for wi_r, wi_i in zip( w_r, w_i )) )
        liftedWeightsConj = top // bottom
        lhs = liftedWeightsConj * steeringVectorParam
        rhs = np.array( [[1.0], [0.0]] )
        P.add_constraint( lhs == rhs )

        P.set_objective( "min", weights.T * csmParam * weights )
        P.solve( verbose=0 )

        # We extract out the real and imaginary parts from our weight vector and return it as a
        # complex 1D array.
        return np.array( weights.value[0:numberElements] +
                        1j * weights.value[numberElements:] ).reshape( numberElements  )


        # Below is the original implementation we had for this algorithm using cvxpy + cvxopt.
        # Modeling the problem was more straightforward given that it better handled the quadratic
        # objective defined in terms of complex vectors and matrices. However, it was slow, and it
        # seemed in capable of handling the robust weight computuation optimization problem.
        #
        # It also had some stability issues, namely it would crap out with a dimension mismatch
        # error due to some internal bug in cvxpy's layer that translates from cvxopt output to a
        # standardized form. For these reasons, we looked into other solvers capable of handling
        # both complex valued vectors and matrices as well as quadratic objectives.
        #
        # This led us to finding this (https://peterwittek.com/sdp-in-python.html) site that
        # recommended Picos and the cvxpy's list of supported solvers and their capabilities
        # (https://cvxgrp.github.io/cvxpy/tutorial/advanced/index.html#solve-method-options) that
        # mentioned Mosek as a solver capable of solving up to semidefinite programs. In the end,
        # that's how we arrived at our choice of back-end solver (Mosek) and front-end interface
        # (Picos).

        #numberElements = steeringVector.size
        #steeringVector2D = steeringVector.reshape( (numberElements, 1) )

        #weights = cvx.Variable( (numberElements, 1), complex=True )
        #distortionlessResponseConstraint = [weights.H * steeringVector2D == 1.0]

        # The quad form is given by (w* R w), where w is our vector of weights and R is our sample
        # cross spectral matrix.
        #minimizePowerObjective = cvx.Minimize( cvx.quad_form( weights, csm ) )
        #problem = cvx.Problem( minimizePowerObjective, distortionlessResponseConstraint )
        #problem.solve()

        # Reshape the weights into a 1D array.
        #return weights.value.reshape( numberElements )

class RobustCaponWeights:
    """
    Computes the robust Capon beamformer weights given by:
    Robust Adaptive Beamforming Using Worst-Case Performance Optimization: A Solution to the Signal
    Mismatch Problem

    https://www.researchgate.net/publication/3318526_Robust_adaptive_beamforming_using_worst-case_performance_optimization_A_solution_to_the_signal_mismatch_problem

    The robust Capon beamformer weights address errors in the steering vector by constraining
    response to be at least unity gain within in some uncertainty region centered on the desired
    direction of arrival. It otherwise seeks to minimize the power of sample cross spectral matrix
    just like the standard Capon beamformer.

    """
    def __init__( self, steeringVectorError ):
        self._steeringVectorError = steeringVectorError

    def compute( self, csm, steeringVector ):
        # Reshape the steering vector so that it's viewed as a 2D array. This just keeps things
        # consistent in our model, where our weight vector is actually shaped like a 2D array.
        # We'll be sure to reshape the final weights as a 1D array after the optimization runs.
        numberElements = steeringVector.size
        steeringVector2D = steeringVector.reshape( (numberElements, 1) )

        # Similar to the standard Capon weight algorithm, we have to hoist our complex vectors and
        # matrices as real matrices in order for Picos to properly handle them.
        liftedCsm = np.bmat( [[np.real( csm ), -np.imag( csm )],
                               [np.imag( csm ), np.real( csm )]] )

        liftedSteeringVector = np.vstack( [np.vstack( [np.real( sv ),
                                                       np.imag( sv )] ) for sv in steeringVector2D] )

        P = pic.Problem()
        weights = P.add_variable( "weights", (2 * numberElements, 1) )
        csmParam = pic.new_param( "csm", liftedCsm )
        steeringVectorParam = pic.new_param( 'steeringVector', liftedSteeringVector )
        steeringVectorErrorParamInv = pic.new_param( "steeringVectorError",
                                                    1.0 / self._steeringVectorError )

        # In the robust optimization case, our constraint becomes w*v >= eps * ||w|| + 1. Again, w
        # is the weight vector and v is the steering vector. The value eps here is some known upper
        # bound on the error we can expect in our steering vector, i.e., basically the degree of
        # mismatch. As noted in the above paper, this is actually a second-order cone constraint.
        #
        # When we hoist this to be real-valued, we again lift the weight vector to the vector
        # [Re(weights) ; -Im(weights)]. The right hand side of the constraint technically becomes
        # the vector [ eps * ||w|| + 1 ; 0]; however, it's actually not possible to model this
        # constraint verbatim as a vector inequality using Picos. Three problems arise in doing so.
        #
        # The first is that Picos doesn't like norm expressions being combined with any other affine
        # expression (e.g., multiplication and addition by scalar values). Thus we have to isolate
        # the norm expression to one side of the constraint. This causes our lifted constraint to
        # be transformed into (w'v - [1 ; 0]) * (1 / eps) >= [ ||w|| ; 0 ]. Note here that in the lifted
        # constraint, the conjugate transpose turns into simply the transpose.
        #
        # This immediately creates the second problem with Picos. Namely, it does not like having a
        # matrix where one of its entries is defined by the norm of some variable. We easily work
        # around this by enumerating each of the following constraints explicitly (where [i] denotes
        # the ith quantity in the vector):
        #
        #     ((w'v)[0] - 1) * (1 / eps) >= ||w||
        #     ((w'v)[1] - 0) * (1 / eps) == 0
        #
        # We actually impose equality on the second constraint because in the complex formulation we
        # want the left hand side to be real-valued.
        #
        # The first constraint in the above list is actually a second order cone constraint. The way
        # it is written right now though leads to a third, albeit trivial, problem, and that is that
        # Picos does not define the >= for norm expressions on the right hand side. Instead, as per
        # the library's documentation (http://picos.zib.de/expression.html#picos.Norm), the proper
        # operator for expressing an inequality with a norm expression is with < which is used for
        # expressing a less than *or equal to* constraint. With these issues all resolved, we are
        # then able to model the robust optimization problem using Picos.
        #
        w_r = weights[0:numberElements]
        w_i = weights[numberElements:]
        top = functools.reduce( lambda x, y: x & y, (wi_r & wi_i for wi_r, wi_i in zip( w_r, w_i )) )
        bottom = functools.reduce( lambda x, y: x & y, (-wi_i & wi_r for wi_r, wi_i in zip( w_r, w_i )) )
        liftedWeightsConj = top // bottom
        lhs = steeringVectorErrorParamInv * (liftedWeightsConj * steeringVectorParam - np.array( [[1.0], [0.0]] ))
        P.add_constraint( pic.tools.norm( weights ) < lhs[0] )
        P.add_constraint( lhs[1] == 0.0 )

        P.set_objective( "min", weights.T * csmParam * weights )
        P.solve( verbose=0 )

        # We extract out the real and imaginary parts from our weight vector and return it as a
        # complex 1D array.
        return np.array( weights.value[0:numberElements] +
                        1j * weights.value[numberElements:] ).reshape( numberElements  )

class SlidingWindow:
    @classmethod
    def createWithAccumulator( cls, windowHalfWidth, initialBlocks, onSlideCb, createAccumulatorCb ):
        assert createAccumulatorCb, "Callback for creating an accumulator must be provided."
        window = cls( windowHalfWidth, initialBlocks, onSlideCb )
        accumulator = createAccumulatorCb( window )
        return (window, accumulator)

    def __init__( self, windowHalfWidth, initialBlocks, onSlideCb ):
        assert windowHalfWidth > 0
        self._windowWidth = 2 * windowHalfWidth + 1

        assert initialBlocks is not None
        numberInitialBlocks = initialBlocks.shape[0]
        assert initialBlocks.ndim > 1, "Initial blocks must be given as an N-D array with N > 1."
        assert numberInitialBlocks <= self._windowWidth, \
            "Number of initial blocks must be less than or equal to window width %d" % self._windowWidth

        # Initialize the window with the initial blocks we were given, zeroing out the remainder of
        # the blocks.
        iFirstInitialBlock = self._windowWidth - numberInitialBlocks
        self.BlockShape = initialBlocks.shape[1:]
        self.Window = np.empty( (self._windowWidth,) + self.BlockShape )
        self.Window[0:iFirstInitialBlock, :] = np.zeros( self.BlockShape )
        self.Window[iFirstInitialBlock:, :] = initialBlocks

        self._iFirst = 0
        self._iCurrent = windowHalfWidth
        self._iLast = self._windowWidth - 1
        self._onSlide = onSlideCb

    @property
    def FirstBlock( self ):
        return self.Window[self._iFirst, :]

    @property
    def CurrentBlock( self ):
        return self.Window[self._iCurrent, :]

    @property
    def LastBlock( self ):
        return self.Window[self._iLast, :]

    def slideWindow( self, accumulator, newBlock=None ):
        if newBlock is not None:
            assert newBlock.shape == self.BlockShape, "Input block must have shape %s" % self.BlockShape
        assert self._iCurrent != self._iLast, "Cannot slide window past last block stored."

        # Callback so that callers have a chance to update the accumulator.
        if self._onSlide:
            self._onSlide( self, newBlock, accumulator )

        # Slide the first and current block indices forward.
        self._iFirst = self._advanceIndex( self._iFirst )
        self._iCurrent = self._advanceIndex( self._iCurrent )

        if newBlock is not None:
            # Slide the last index forward and write the new block to the new end of our window.
            self._iLast = self._advanceIndex( self._iLast )
            self.Window[self._iLast, :] = newBlock

    def _advanceIndex( self, index ):
        return (index + 1) % self._windowWidth

def createRunningSum( slidingWindow ):
    return np.sum( slidingWindow.Window, axis=0 )

def updateRunningSum( slidingWindow, newBlock, runningSum ):
    runningSum -= slidingWindow.FirstBlock
    if newBlock is not None:
        runningSum += newBlock

def plotArrayGeometry( geometry,
                      plotSize=(5.0, 8.0),
                      figureTitle=None ):
    fig = plt.figure( figsize=plotSize, dpi=90 )
    ax = fig.add_subplot( 111 )

    showGrid = True
    figureTitle = figureTitle or "Array Geometry"
    _ = ax.plot( geometry.X, geometry.Y, "o", markersize=7 )
    _ = ax.grid( showGrid, which="both" )
    _ = ax.set_title( figureTitle )

def plotTimeSeries( etsFileName,
                   numberSeconds=1.0,
                   plotsPerRow=4,
                   plotSize=(4.0, 2.0),
                   figureTitle=None,
                   plotTitleFormatter=None ):
    with ElementTimeSeries.open( etsFileName ) as etsFile:
        numberSamples = int( np.ceil( numberSeconds * etsFile.SamplingRate ) )
        sampleTimes = np.linspace( 0.0, numberSeconds, numberSamples, endpoint=False )
        samples = np.empty( (numberSamples, etsFile.NumberElements) )
        samplesRead = 0

        while samplesRead < numberSamples:
            snapshot = etsFile.readSnapshot()
            if snapshot.size == 0:
                break
            samplesToRead = np.minimum( numberSamples - samplesRead, snapshot.shape[0] )
            samples[samplesRead:(samplesRead + samplesToRead), :] = snapshot[0:samplesToRead, :]
            samplesRead += samplesToRead

        sampleTimes = sampleTimes[0:samplesRead]

        plotWidth, plotHeight = plotSize
        numberRows = int( np.ceil( etsFile.NumberElements / plotsPerRow ) )
        fig, ax = plt.subplots( numberRows, plotsPerRow,
                               figsize=(plotWidth * plotsPerRow, numberRows * plotHeight),
                               dpi=90,
                               squeeze=False )

        # Disable those subplots that aren't actually going to be used.
        iUnusedAxes = np.unravel_index( [i for i in range( etsFile.NumberElements, ax.size )],
                                         ax.shape )
        for unusedAxis in ax[iUnusedAxes]:
            unusedAxis.set_visible( False )

        title = figureTitle or "Element Time Series"
        fig.suptitle( title, y=1, fontsize=14 )
        fig.tight_layout( pad=4, h_pad=2, w_pad=2 )

        for iElement in range( etsFile.NumberElements ):
            axRow = int(iElement / plotsPerRow)
            axCol = int(iElement - axRow * plotsPerRow)
            plotTitle = (plotTitleFormatter( iElement ) if plotTitleFormatter
                         else ("Element %d" % (iElement + 1)))

            _ = ax[axRow, axCol].plot( sampleTimes, samples[:, iElement] )
            _ = ax[axRow, axCol].set_title( plotTitle )
            _ = ax[axRow, axCol].grid( True, which="both" )

def plotPowerSpectra( fftFileName,
                     numberSnapshots=1,
                     band=FrequencyBand(),
                     plotsPerRow=2,
                     plotSize=(6.0, 2.5),
                     figureTitle=None,
                     figureTitleFormatter=None,
                     plotTitleFormatter=None ):
    with FourierSpectra.open( fftFileName ) as fftFile:
        binFrequencies = fftFile.BinFrequencies
        iStartBin, iStopBin = getBinIndicesForBand( band, binFrequencies )
        binFrequencies = binFrequencies[iStartBin:iStopBin]

        for iSnapshot in range( numberSnapshots ):
            spectra = fftFile.readSnapshot()
            if spectra.size == 0:
                break
            spectra = spectra[iStartBin:iStopBin, :]

            plotWidth, plotHeight = plotSize
            numberRows = int(np.ceil( fftFile.NumberChannels / plotsPerRow ))
            fig, ax = plt.subplots( numberRows, plotsPerRow,
                                   figsize=(plotWidth * plotsPerRow, plotHeight * numberRows),
                                   dpi=90,
                                   squeeze=False )

            # Disable those subplots that aren't actually going to be used.
            iUnusedAxes = np.unravel_index( [i for i in range( fftFile.NumberChannels, ax.size )],
                                             ax.shape )
            for unusedAxis in ax[iUnusedAxes]:
                unusedAxis.set_visible( False )

            if not figureTitle:
                figureTitle = (figureTitleFormatter( iSnapshot ) if figureTitleFormatter
                               else ("Fourier Spectra Snapshot %d" % (iSnapshot + 1)))

            fig.suptitle( figureTitle, y=1, fontsize=14 )
            fig.tight_layout( pad=4, h_pad=3, w_pad=4 )

            for iChannel in range( fftFile.NumberChannels ):
                axRow = int(iChannel / plotsPerRow)
                axCol = int(iChannel - axRow * plotsPerRow)
                channelPower = np.absolute( spectra[:, iChannel] ) ** 2 / fftFile.FftLength
                channelDb = 10.0 * np.log10( channelPower )
                channelTitle = (plotTitleFormatter( iChannel ) if plotTitleFormatter
                                else ("Channel %d" % (iChannel + 1)))

                _ = ax[axRow, axCol].plot( binFrequencies, channelDb )
                _ = ax[axRow, axCol].set_title( channelTitle )
                _ = ax[axRow, axCol].grid( True, which="both" )
                _ = ax[axRow, axCol].set_ylabel( "Frequency (Hz)" )
                _ = ax[axRow, axCol].set_ylabel( "Power (dB)" )

if __name__ == "__main__":
    import time
    import warnings

    warnings.filterwarnings( "ignore", module="matplotlib" )

    # This is a flag used to quickly toggle off processing so that we don't regenerate all data.
    process = True

    #####
    # In this section we will be deriving parameters for our array based on more-or-less good rules
    # of thumb. There's a lot more analysis and rigor that goes on in optimal array design, but
    # we're not terribly concerned about that here.
    #

    # Speed of sound in water. This actually varies based on a number of factors, but this number
    # is adequate for our purposes.
    speedOfSound = 1480.0

    # The "design frequency" of array. This is the frequency we typically want to focus on in our
    # downstream processing. However, we didn't see any specific rules of thumb on deciding on how
    # to pick this. For example, since we're typically interested in a range of frequencies, do we
    # perhaps pick half the bandwidth? Or maybe we instead design for the frequency of interest
    # that has the longest wavelength? For our purposes, we arbitrarily picked 400 Hz.
    #
    # From the target frequency, we're able to readily compute our target wavelength lambda.
    targetFrequency = 400.0
    targetLambda = speedOfSound / targetFrequency

    # Spacing between the elements in our array and the desired length of the array as functions of
    # the target wavelength.
    lambdaElementSpacing = 0.5
    lambdaArrayLength = 10.0

    # Required number of elements needed to achieve at least our desired length.
    elementSpacing = targetLambda * lambdaElementSpacing
    numberElements = int( np.ceil( lambdaArrayLength / lambdaElementSpacing ) + 1 )

    # Nyquist rate is the minimum sampling rate needed in order to accurately sample a signal
    # without introducing aliasing errors. It must be at least twice the frequency of the signal of
    # interest.
    nyquistRate = 2.0 * targetFrequency

    # Sampling rate of each element in our array. We want a sampling rate that is several times
    # greater than our Nyquist rate.
    # http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/SonarBeamforming/lecture12/discrete.html
    nyquistFactor = 10.0
    samplingRate = nyquistFactor * nyquistRate

    # For convenience, we set our snapshot duration (in seconds) to be equal to the size of our FFT
    # used during beamforming.
    secondsPerSnapshot = 10.0

    #####
    # In this next section, we construct our simulated array using the design parameters computed
    # above. Using this, we simulate the signals of a couple of targets at different directions
    # relative to the array along with some Gaussian white noise. This simulated time series data
    # will be the input into our beamforming process that occurs later.
    #

    # Create a uniform linear array, which means all elements are colinear, and equal spacing is
    # between adjacent elements.
    geometry = ArrayGeometry.createUniformLinear( numberElements, elementSpacing )

    # Each element is represented by our canoncial hydrophone. The hydrophone is responsible for
    # converting pressure from acoustic waves into a corresponding voltage that is sampled as a
    # function of time. The discrete voltage signal becomes our digital representation of the
    # acoustic signals in our simulated array environment. In our array, we use identical,
    # isotropic hydrophones, which means their response to acoustic waves is equal in all
    # directions.
    hydrophone = Hydrophone()

    # Using the pieces we've previously constructed, we now construct our simulated array.
    arraySim = ArraySimulator( geometry,
                              hydrophone,
                              samplingRate,
                              secondsPerSnapshot,
                              speedOfSound )

    #####
    # Below we will construct our simulated environment by adding a white noise generator to
    # simulate ocean noise and different target signals the simulated array will record in its
    # generated time series.
    #

    # Add a white noise generator that will simulate ambient ocean noise. 100 dB is approximate
    # level for ambient noise at sea state 4.
    # Reference: http://www.arc.id.au/SoundLevels.html
    #
    # We set a constant seed so that we produce the same data with each run.
    #
    noiseGenerator = WhiteNoiseGenerator.createFromSoundPressureLevel( soundPressureLevel=100,
                                                                      hydrophone=hydrophone )
    noiseGenerator.setSeed( 1 )
    arraySim.setNoiseGenerator( noiseGenerator )

    # Add one target 30 degrees to the right of broadside. This will be our primary target and will
    # have a sawtooth acoustic signature. The sawtooth signal generates a set of harmonics whose
    # amplitude falls off at a rate of 1/N. This will help us better identify our signal in our
    # Fourier spectra.
    #
    # 120 dB is about one order of magnitude less than the sound of a fin whale call 100 meters away.
    # http://www.arc.id.au/SoundLevels.html
    #
    primaryTarget = Target( PositionAzEl( azimuth=-30.0 ),
                           frequency=60.0,
                           soundPressureLevel=120,
                           signalGenerator=SawtoothGenerator() )
    arraySim.addTarget( primaryTarget )

    # We add another target 45 degrees to the right of broadside, It is louder than our primary
    # target and will have a tendency to dominate. It is placed 6 dB louder than our primary target,
    # which should represent roughly twice the amount of power. Its acoustic signature will be a
    # sine wave, so it will have a single peak in the Fourier spectra at its configured frequency.
    arraySim.addTarget( Target( PositionAzEl( azimuth=-45.0 ),
                               frequency=100.0,
                               soundPressureLevel=126,
                               signalGenerator=SineGenerator() ) )

    #####
    # Next we configure our transformer, which is responsible for converting our element time series
    # data in a set of Fourier spectra, one for each element in the array.
    #
    # The transformer accomplishes this by means of a fast Fourier transform. Two common parameters
    # used in configuring this stage are a windowing function and an overlap percentage.
    #
    # The discrete Fourier transform assumes the signal transformed is periodic. However, since the
    # transformer works on snapshots of time series data at a time, it's unlikely the samples in the
    # FFT buffer at any given time resemble a periodic signal. The windowing function is used to
    # ensure the samples at the beginning and end of our FFT buffer approach a common value, thus
    # allowing them to appear periodic if you were to stack them end-to-end.
    #
    # The transformer can be configured to keep some percentage of the samples from the previous
    # transform in the FFT buffer for the next transform. This is controlled by the overlap
    # percentage.
    #
    # One reason we might care about setting the overlap percentage to some non-zero value is when
    # it's used in conjunction with windowing. In particular, if we use a Hann window with an
    # overlap percentage of 50%, it can be shown that our Fourier spectra won't exhibit amplitude
    # fluctuations that differ from our original time series data.
    # Reference:
    # https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
    #
    transformer = FourierTransformer( overlap=0.5, window=HannWindow() )

    #####
    # Finally we configure our beamformer. Fundamentally, a beamformer requires only a few items:
    # the geometry of the array being processed, a list of output beams to form, and some
    # environmental information such as the speed of sound in water.
    #
    # A broadband beamformer in the frequency domain typically works by using the FFT to break a
    # broadband signal into a number of narrowband components (the individual frequency bins of the
    # FFT's output) and performs narrowband beamforming on each of those components.
    #
    # Since the transformed time series data can have A LOT of frequency bins (depending on the size
    # of the transform), the amount of work the beamformer must perform can quickly become huge.
    #
    # For this reason, the beamformer we developed can also take in an optional frequency band that
    # tells the beamformer over which frequency components to process. If no frequency band were
    # specified, then the beamformer would work over the entire bandwidth available in the incoming
    # Fourier spectra data stream.
    #
    # There are two other key processing parameters that impact the performance of the beamformer.
    # The first is a time averaging factor. Most beamforming algorithm formulations assume the true
    # cross spectral matrix is known for both the signal of interest and the noise in the
    # environment. In practice this isn't true, and the cross spectral matrix is replaced by the
    # sample estimate of the CSM.
    #
    # If X(f, k) is the spectral data for frequency component f and snapshot k, then the sample CSM
    # is the average of X(f,k) X*(f,k) for k = k1 to kJ. Here J is the number of snapshots that are
    # averaged to form our sample CSM. In practice, it's important to choose a high enough value for
    # J so that we are getting an accurate estimate for our true CSM.
    #
    # The second key processing parameter is the algorithm used for computing the beamforming
    # weights. The beamforming weights are computed on per frequency, per output beam basis. They
    # essentially alter the response of the beamformer and steer maximum response towards signals
    # and directions we care about while placing nulls (areas of little to no response) towards
    # interfering signals and directions.
    #
    # Different algorithms have different trade offs in terms of computational complexity and
    # performance. In general, conventional weights are the fastest to compute, but they offer no
    # adaptivity against interfering signals. Different adaptive algorithms try to pick optimal
    # weights subject to different response constraints. Different algorithms also have different
    # degrees of sensitivty to error in both the sample CSM and the steering vector formed for a
    # particular output beam.
    #

    # The output beams we wish to form. Each channel in our beamformer's output will correspond to
    # one of these output beams. Each channel's spectrum will show the frequency content arriving
    # from that beam's direction.
    #
    # Note that these beam angles are relative to our local coordinate system and not to our array.
    # We have ensured that our array is positioned in our coordinate system such that its broadside,
    # which is the direction perpendicular to the axis of the array, coincides with 0 degrees.
    # It is customary in sonar applications to have broadside labeled as 0 degrees.
    #
    outputBeams = (Beam( 90.0 ),
                   Beam( 60.0 ),
                   Beam( 45.0 ),
                   Beam( 30.0 ),
                   Beam( 0.0 ),
                   Beam( -30.0 ),
                   Beam( -45.0 ),
                   Beam( -60.0 ),
                   Beam( -90.0 ))

    # The frequency band we want to process in our beamformer. A real world application would have
    # a far greater frequency extent processed, but for purposes of keeping computation time down,
    # we limit our processing band. This will at least enable us to see our two simulated targets in
    # our beamformer's output.
    frequencyBandToProcess = FrequencyBand( startHertz=50.0, stopHertz=130.0 )

    # Here we choose a snapshot average count of twice the number of elements. This adheres to the
    # so called RMB rule that says at least 2N snapshots should be used in order to keep the average
    # loss to less than 3 dB.
    # Reference:
    # I. S. Reed, J. D. Mallett and L. E. Brennan, "Rapid Convergence Rate in Adaptive Arrays"
    beamformer = Beamformer( geometry, outputBeams, frequencyBandToProcess,
                            snapshotAverageCount=2 * geometry.NumberElements,
                            speedOfSound=speedOfSound )

    #####
    # Below we now run all parts of our system end to end. We perform beamforming using different
    # weight algorithms so that we can compare their performance afterward.
    #

    # Setup paths to the files we will be generating at each stage of the system.
    outputFolder = pathlib.Path( pathlib.Path.home(), "convexAbf" )
    outputFolder.mkdir( parents=True, exist_ok=True )

    etsFileName = str( outputFolder / "rawTimeSeries.ets" )
    elementFftFileName = str( outputFolder / "elementSpectra.fft" )
    conventionalBeamformedFftFileName = str( outputFolder / "beamformedSpectraConventional.fft" )
    caponBeamformedFftFileName = str( outputFolder / "beamformedSpectraCapon.fft" )
    robustCaponBeamformedFftFileName = str( outputFolder / "beamformedSpectraRobustCapon.fft" )

    # Generate our simulated time series. We arbitrarily picked how many snapshots of time series
    # data we'll generate.
    print( "Simulating time series..." )
    if process:
        arraySim.simulate( numberSnapshots=30, fileName=etsFileName )

    # Produce the Fourier spectra of our time series.
    print( "Computing spectral information..." )
    if process:
        transformer.transformTimeSeries( etsFileName, elementFftFileName )

    # Run the beamformer for each of the following weight algorithms:
    #
    #     Conventional weights - no adaptivity
    #     Capon weights - standard minimum variance distortionless response (MVDR) weighting
    #     Robut Capon weights - MVDR with uncertainty modelled in the steering vector
    #
    print( "Forming output beams with conventional beamformer..." )
    startTime = time.time()
    if process:
        beamformer.process( elementFftFileName,
                           conventionalBeamformedFftFileName,
                           ConventionalWeights() )
    conventionalTime = time.time() - startTime

    print( "Forming output beams with standard Capon beamformer..." )
    startTime = time.time()
    if process:
        beamformer.process( elementFftFileName,
                           caponBeamformedFftFileName,
                           CaponWeights() )
    caponTime = time.time() - startTime

    print( "Forming output beams with robust Capon beamformer..." )
    startTime = time.time()
    if process:
        beamformer.process( elementFftFileName,
                           robustCaponBeamformedFftFileName,
                           RobustCaponWeights( steeringVectorError=0.1 ) )
    robustCaponTime = time.time() - startTime

    #####
    # In this section we output all the results of the above simulations. We display information on
    # the array we designed, the data we generated, and we produce plots of the element time series,
    # element Fourier spectra, and beamformed Fourier spectra for each of the beamformers.
    #

    print( "Array design frequency: %g Hz" % targetFrequency )
    print( "Array design wavelength: %g meters" % targetLambda )
    print( "Array design element spacing as function of lambda: %g meters" % lambdaElementSpacing )
    print( "Array design length as function of lambda: %g meters" % lambdaArrayLength )
    print( "Computed element spacing: %g meters" % elementSpacing )
    print( "Computed number array elements: %d" % numberElements )
    print( "Computed sampling rate: %g Hz" % samplingRate )
    print( "Seconds per snapshot: %g" % secondsPerSnapshot )
    print( "Speed of sound: %g m/s" % speedOfSound )
    print( "Conventional beamforming time: %g sec" % conventionalTime )
    print( "Standard Capon beamforming time: %g sec" % caponTime )
    print( "Robust Capon beamforming time: %g sec" % robustCaponTime )

    plotArrayGeometry( geometry )

    # Plot only two cycles of our primary target's signature. Note that this will be the
    # superposition of the primary target along with all other targets and whatever noise was in the
    # simulation.
    plotTimeSeries( etsFileName,
                   numberSeconds = 2.0 / primaryTarget._frequency,
                   figureTitle="Element time series for first simulated snapshot" )

    plotPowerSpectra( elementFftFileName,
                     band=FrequencyBand( 50.0, 130.0 ),
                     figureTitle="Fourier spectra for first simulated snapshot" )

    plotPowerSpectra( conventionalBeamformedFftFileName,
                     figureTitle="Conventionally beamformed spectra for first simulated snapshot",
                     plotTitleFormatter=lambda iBeam: "Beam %d" % (iBeam + 1) )

    plotPowerSpectra( caponBeamformedFftFileName,
                     figureTitle="Capon beamformed spectra for first simulated snapshot",
                     plotTitleFormatter=lambda iBeam: "Beam %d" % (iBeam + 1) )

    plotPowerSpectra( robustCaponBeamformedFftFileName,
                     figureTitle="Robust Capon beamformed spectra for first simulated snapshot",
                     plotTitleFormatter=lambda iBeam: "Beam %d" % (iBeam + 1) )

## This test demonstrates how a denormal number time[5] fed into sig.sawtooth produces a NaN ##
#snapshotStartTime = 0.0
#snapshotEndTime = 10.0
#samplesPerSnapshot = 80000
#targetTimeDelay = 0.0006250000000000004
#snapshotTime = np.linspace( snapshotStartTime, snapshotEndTime, samplesPerSnapshot, endpoint=False )
#frequency = 60.0
#time = snapshotTime - targetTimeDelay
#width = 1.0
#samples = sig.sawtooth( 2.0 * np.pi * frequency * time, width )

## TODO -- random things to consider doing ##
# * Low pass filter generated signals so that they are band limited.
#   * Make this a property of the Hydrophone class.
#   * Or maybe a property of ArraySimulator (basically a block anti-alias filter)

## LINKS -- these are links I've piled up but haven't sifted through which are useful... ##
## some of these may be redundant with ones sprinkled in the code above ##
#
# https://en.wikipedia.org/wiki/Plane_wave_expansion
# https://stackoverflow.com/questions/4364823/how-do-i-obtain-the-frequencies-of-each-value-in-an-fft
# https://dsp.stackexchange.com/questions/40766/calculating-values-of-frequency-bins-in-python
# https://www.electronics-tutorials.ws/accircuits/complex-numbers.html
# http://www.labbookpages.co.uk/audio/beamforming/delayCalc.html
# https://en.wikipedia.org/wiki/Sensor_array
# https://www.mathworks.com/help/phased/ug/uniform-linear-array.html
# https://www.mathworks.com/help/phased/ref/phased.isotropichydrophone-system-object.html
# https://en.wikipedia.org/wiki/Root_mean_square
# http://www.indiana.edu/~emusic/etext/acoustics/chapter1_amplitude4.shtml
# https://en.wikipedia.org/wiki/Sine_wave
# https://electronics.stackexchange.com/questions/96205/how-to-convert-volts-in-db-spl
# https://en.wikipedia.org/wiki/Decibel#Voltage
# https://geoffthegreygeek.com/microphone-sensitivity/
# http://rug.mnhn.fr/seewave/HTML/MAN/micsens.html
# https://support.biamp.com/General/Audio/Microphone_sensitivity
# http://www.cetaceanresearch.com/hydrophones/c75-hydrophone/index.html
# https://sensortechcanada.com/custom-hydrophones/broadband-hydrophones/
# http://hydrophoneguide.com/what-is-a-hydrophone-2/
# https://fas.org/man/dod-101/sys/ship/acoustics.htm
# https://oceanexplorer.noaa.gov/explorations/sound01/background/acoustics/acoustics.html
# http://www.arc.id.au/SoundLevels.html
# http://gentleseas.blogspot.com/2016/10/submarine-noise.html
# http://armscontrol.ru/subs/snf/snf03221.htm
# https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
# https://fas.org/man/dod-101/navy/docs/es310/uw_acous/uw_acous.htm
# https://en.wikipedia.org/wiki/Array_processing
# https://en.wikipedia.org/wiki/Phased_array_ultrasonics
# https://en.wikipedia.org/wiki/Direction_of_arrival
# https://en.wikipedia.org/wiki/Beamforming
# https://en.wikipedia.org/wiki/Window_function
# https://en.wikipedia.org/wiki/Sensor_array
# https://www.spectraplus.com/DT_help/fft_size.htm
# http://tiao.io/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
# https://stackoverflow.com/questions/39269804/fft-normalization-with-numpy
# https://en.wikipedia.org/wiki/Wavenumber
# https://link.springer.com/chapter/10.1007%2F978-3-642-25905-0_2