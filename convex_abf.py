# https://www.cs.ccu.edu.tw/~wtchu/courses/2012s_DSP/Lectures/Lecture%203%20Complex%20Exponential%20Signals.pdf
# http://www.labbookpages.co.uk/audio/beamforming/delayCalc.html
# https://skynet.ee.ic.ac.uk/papers/2011_C_MathsInDefence_VirtualLinearArray.pdf
# https://en.wikipedia.org/wiki/Sensor_array
# http://www.comm.utoronto.ca/~rsadve/Notes/ArrayTheory.pdf
# https://research.ijcaonline.org/volume61/number11/pxc3884758.pdf
# http://www.personal.psu.edu/faculty/m/x/mxm14/sonar/beamforming.pdf
# http://www.uio.no/studier/emner/matnat/ifi/INF5410/v12/undervisningsmateriale/foils/AdaptiveBeamforming.pdf

import itertools
import pathlib
import struct
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig

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
        return cls( dataFileName, cls.MODE_WRITE, samplesPerSnapshot, numberElements,
                   samplingRate, sampleSizeBytes )

    @classmethod
    def open( cls, dataFileName ):
        return cls( dataFileName, cls.MODE_READ )

    def writeSnapshot( self, snapshot ):
        numberSnapshotSamples, numberSnapshotElements = snapshot.shape
        assert numberSnapshotSamples == self.SamplesPerSnapshot, \
            "Number of samples per snapshot must be %d" % self.SamplesPerSnapshot
        assert numberSnapshotElements == self.NumberElements, \
            "Number of elements in snapshot must be %d" % self.NumberElements

        self._dataFile.write( snapshot.tobytes() )

    def readSnapshot( self ):
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
    data. Each snapshot contains M frequency bins by N channels many complex128 values. The value
    M is the integer given by floor( FftLength / 2 ) + 1.

    The global header contains the following values in order:
        FftLength      (64-bit unsigned int)
        NumberChannels (64-bit unsigned int)
        SamplingRate   (64-bit floating point)

    """
    MODE_READ = 0
    MODE_WRITE = 1

    @classmethod
    def create( cls, fftLength, numberChannels, samplingRate, dataFileName ):
        return cls( dataFileName, cls.MODE_WRITE, fftLength, numberChannels, samplingRate )

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
        self.NumberChannels = self._readFieldFromFile( self.NumberChannelsHeaderFormat,
                                                      self.NumberChannelsHeaderSize )
        self.SamplingRate = self._readFieldFromFile( self.SamplingRateHeaderFormat,
                                                   self.SamplingRateHeaderSize )

    def _readFieldFromFile( self, fieldFormat, fieldSizeBytes ):
        # Note: struct.unpack returns a tuple even if there's only a single element.
        return struct.unpack( fieldFormat, self._dataFile.read( fieldSizeBytes ) )[0]

    def _writeHeader( self ):
        self._writeFieldToFile( self.FftLength, self.FftLengthHeaderFormat )
        self._writeFieldToFile( self.NumberChannels, self.NumberChannelsHeaderFormat )
        self._writeFieldToFile( self.SamplingRate, self.SamplingRateHeaderFormat )

    def _writeFieldToFile( self, fieldValue, fieldFormat ):
        self._dataFile.write( struct.pack( fieldFormat, fieldValue ) )

    def __init__( self, dataFileName, mode, fftLength=None, numberChannels=None, samplingRate=None ):
        assert mode in (self.MODE_READ, self.MODE_WRITE)
        assert dataFileName

        if mode == self.MODE_WRITE:
            assert fftLength > 0 and type(fftLength) is int, \
                "FFT length must be positive integer."
            assert numberChannels > 0 and type(numberChannels) is int, \
                "Number of channels must be positive integer."
            assert samplingRate > 0, "Sampling rate must be positive."

        self._dataFileName = dataFileName
        self._dataFile = None
        self._mode = mode
        self.FftLength = fftLength
        self.NumberChannels = numberChannels
        self.SamplingRate = samplingRate
        self._binDtype = np.dtype( np.complex128 )

    @property
    def NumberBins( self ):
        return int(np.floor( self.FftLength / 2 ) + 1) if self.FftLength else None

    @property
    def BinResolution( self ):
        return self.SamplingRate / self.FftLength

    @property
    def BinFrequencies( self ):
        numberBins = self.NumberBins
        return np.arange( numberBins ) * self.BinResolution if numberBins else np.array( [] )

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
    FftLengthHeaderSize = 8
    NumberChannelsHeaderSize = 8
    SamplingRateHeaderSize = 8

    FftLengthHeaderFormat = "=Q"
    NumberChannelsHeaderFormat = "=Q"
    SamplingRateHeaderFormat = "=d"

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
        polar = np.radians( sourcePosition.PolarAngle )
        azimuth = np.radians( sourcePosition.Azimuth )
        sinElevation = np.sin( polar )
        sinAzimuth = np.sin( azimuth )
        cosElevation = np.cos( polar )
        cosAzimuth = np.cos( azimuth )

        xA = sinElevation * cosAzimuth
        yA = sinElevation * sinAzimuth
        zA = cosElevation

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

            with FourierSpectra.create( fftLength, numberChannels, samplingRate, fftFileName ) as outputFft:
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
    def __init__( self, azimuth, elevation ):
        self.Azimuth = azimuth
        self.Elevation = elevation

    @property
    def PolarAngle( self ):
        return elevationToPolarAngle( self.Elevation )

def elevationToPolarAngle( elevation ):
    return 90.0 - elevation

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

class Beamformer:
    def __init__( self, arrayGeometry, outputBeams, snapshotAverageCount, speedOfSound ):
        assert snapshotAverageCount > 0

        self._arrayGeometry = arrayGeometry
        self._outputBeams = outputBeams
        self._snapshotAverageCount = snapshotAverageCount
        self._speedOfSound = speedOfSound

    def process( self, inputFileName, outputFileName ):
        with FourierSpectra.open( inputFileName ) as inputFft:
            # First compute the optimal snapshot buffer size.
            numberBufferedSnapshots = self._computeNumberBufferedSnapshots( inputFft.NumberBins )

            #The motivation for this is that
            # broadband beamforming is both memory and compute bound. If we were to store all
            steeringVectors = self._computeSteeringVectors( inputFft )

            with FourierSpectra.create( inputFft.FftLength, len(self._outputBeams),
                                       inputFft.SamplingRate, outputFileName ) as outputFft:
                pass

    def _computeNumberBufferedSnapshots( self, numberBins ):
        """
        Computes the number of input FFT snapshots to read and buffer at a time.

        The motivation for this is that broadband beamforming where we take an FFT of a signal and
        perform narrowband beamforming on the individual frequency components is both memory and
        compute bound, but the former will be a far greater limiting factor.

        To illustrate, let N be the number of input elements, M the number of frequency bins, Q the
        number of output beams, and J the number of snapshots we need to average to build our CSM.
        In order to keep our beamformer output within 3 dB of the optimal SINR, we need to ensure
        the number of snapshots we use for computing the cross spectral matrix is at least twice
        the number of elements.

        Consider that to compute the weights for a single frequency for a single output beam, we
        need NQ + MNJ + MQ many complex values stored in memory. For the next beam or frequency
        beam, we need to recompute the NQ many steering vectors. Thus for one output snapshot,
        there are M many steering vector calculations that must be constantly recomputed every
        snapshot.

        Alternatively, we can precompute the steering vectors upfront and reuse them when computing
        the weights for each snapshot. However, this requires MNQ + MNJ + MQ many complex values in
        memory. Consider a 21 element array steered in 100 different directions with sampled time
        series spanning a 4 Khz bandwidth. If the frequency domain spectrum for this time series has
        0.1 Hz bin resolution, we have at least 40,001 bins. Thus the broadband beamformer requires
        almost 1.9 GB of memory (assuming each complex value comprises two 64-bit doubles).

        It is obvious there's a tension between being memory efficient and being compute efficient.
        We note that if we read and buffer J' snapshots (J' > J), we can begin to amortize the cost
        of recomputing steering vectors. This is because for each block of J' snapshots we read in,
        we can compute the steering vectors for a single (frequency, beam) pair and use them to
        calculate the corresponding weights for J' - J output snapshots before needing to recompute
        the steering vectors for either the next beam or next frequency.

        This strategy requires storing NQ + MNJ' + MQ(J' - J + 1) complex values in memory. The
        breakeven point is the largest value of J' that results in less memory consumption than
        the full precomputation method. However, rarely do we want to choose this for J'; othewise,
        we gain no benefit since we will consume the same order of memory for a more computationally
        expensive process. Instead we may constrain J' to be no larger than some memory budget,
        which can be readily solved.

        """
        pass

    def _computeSteeringVectors( self, inputFft ):
        numberElements = self._arrayGeometry.NumberElements
        numberDirections = len(self._outputBeams)
        numberBins = inputFft.NumberBins
        steeringVectors = np.empty( shape=(numberBins, numberElements, numberDirections),
                                   dtype=np.complex )

## OUTPUT SETUP ##

# Dump all output to a temp file location on the file system.
outputFolder = pathlib.Path( tempfile.gettempdir(), 'convexAbf' )
outputFolder.mkdir( parents=True, exist_ok=True )
etsFileName = str( outputFolder / 'array.ets' )
elementFftFileName = str( outputFolder / 'element.fft' )

##  ARRAY DESIGN ##
speedOfSound = 1480.0

# NOTE: this may change because typically we're interested in a range of frequencies.
# Maybe half the bandwidth? Maybe design for longest frequency of interest?
targetFrequency = 400.0
targetLambda = speedOfSound / targetFrequency

# Spacing between elements and desired length of the array as functions of the wavelength lambda.
lambdaElementSpacing = 0.5
lambdaArrayLength = 10.0

# Required number of elements needed to achieve at least our desired length.
elementSpacing = targetLambda * lambdaElementSpacing
numberElements = int( np.ceil( lambdaArrayLength / lambdaElementSpacing ) + 1 )
geometry = ArrayGeometry.createUniformLinear( numberElements, elementSpacing )

# Nyquist rate is the minimum sampling rate needed in order to accurately sample a signal without
# introducing aliasing errors. It must be at least twice the frequency of the signal of interest.
# This quantity is called the Nyquist frequency.
nyquistRate = 2.0 * targetFrequency

# Sampling rate of each element in our array. We want a sampling rate that is several times greater
# than our Nyquist rate.
# http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/SonarBeamforming/lecture12/discrete.html
nyquistFactor = 10.0
samplingRate = nyquistFactor * nyquistRate

# For convenience, we set our snapshot duration to be the size of our FFT used during beamforming.
secondsPerSnapshot = 10.0

## ARRAY SIMULATION ##
hydrophone = Hydrophone()
arraySim = ArraySimulator( geometry, hydrophone, samplingRate, secondsPerSnapshot, speedOfSound )

# 100 dB is approximate level for ambient noise at sea state 4.
# http://www.arc.id.au/SoundLevels.html
noiseGenerator = WhiteNoiseGenerator.createFromSoundPressureLevel( soundPressureLevel=100,
                                                                  hydrophone=hydrophone )
noiseGenerator.setSeed( 1 )
arraySim.setNoiseGenerator( noiseGenerator )

# 120 dB is about one order of magnitude less than the sound of a fin whale call 100 meters away.
# For demo purposes, this is ideal as it makes for a quiet target.
# http://www.arc.id.au/SoundLevels.html
primaryTarget = Target( PositionAzEl( azimuth=-30.0 ),
                       frequency=60.0,
                       soundPressureLevel=120,
                       signalGenerator=SawtoothGenerator() )
arraySim.addTarget( primaryTarget )

arraySim.addTarget( Target( PositionAzEl( azimuth=-45.0 ),
                           frequency=100.0,
                           soundPressureLevel=126,
                           signalGenerator=SineGenerator() ) )

# Run the simulation.
numberSnapshots = 30
arraySim.simulate( numberSnapshots, etsFileName )

# Produce the Fourier spectra of our time series.
transformer = FourierTransformer( window=HannWindow() )
transformer.transformTimeSeries( etsFileName, elementFftFileName )

## TESTING ##
import warnings
warnings.filterwarnings( "ignore", module="matplotlib" )

print( "Array design frequency: %g Hz" % targetFrequency )
print( "Array design wavelength: %g meters" % targetLambda )
print( "Array design element spacing as function of lambda: %g meters" % lambdaElementSpacing )
print( "Array design length as function of lambda: %g meters" % lambdaArrayLength )
print( "Computed element spacing: %g meters" % elementSpacing )
print( "Computed number array elements: %d" % numberElements )
print( "Computed sampling rate: %g Hz" % samplingRate )
print( "Seconds per snapshot: %g" % secondsPerSnapshot )
print( "Speed of sound: %g m/s" % speedOfSound )

# Plot the array geometry.
fig = plt.figure( figsize=(5, 8), dpi=90 )
ax = fig.add_subplot( 111 )

showGrid = True
_ = ax.plot( geometry.X, geometry.Y, 'o', markersize=7 )
_ = ax.grid( showGrid, which='both' )
_ = ax.set_title( 'Array Geometry' )

# Plot the first portion of the first snapshot of element time series data captured by all array
# elements in our simulation.
with ElementTimeSeries.open( etsFileName ) as ets:
    plotsPerRow = 3
    numberRows = int(np.ceil( ets.NumberElements / plotsPerRow ))
    fig, ax = plt.subplots( numberRows, plotsPerRow,
                           figsize=(3 * plotsPerRow, numberRows * 1.5), dpi=90,
                           squeeze=False )
    fig.suptitle( 'Element time series for first simulated snapshot', y=1, fontsize=14 )
    fig.tight_layout( pad=4, h_pad=2, w_pad=2 )

    # Plot only two cycles of our primary target's signature. Note that this will be the
    # superposition of the primary target along with all other targets and whatever noise was in the
    # simulation.
    t = np.linspace( 0.0,
                    2.0 / primaryTarget._frequency,
                    2 * int(ets.SamplingRate/ primaryTarget._frequency),
                    endpoint=False )
    s = ets.readSnapshot()

    for iElement in range( ets.NumberElements ):
        axRow = int(iElement / plotsPerRow)
        axCol = int(iElement - axRow * plotsPerRow)
        _ = ax[axRow, axCol].plot( t, s[0:len(t),iElement] )
        _ = ax[axRow, axCol].set_title( 'Element %d' % (iElement + 1) )
        _ = ax[axRow, axCol].grid( True, which='both' )


with FourierSpectra.open( elementFftFileName ) as fft:
    plotsPerRow = 2
    numberRows = int(np.ceil( fft.NumberChannels / plotsPerRow ))
    fig, ax = plt.subplots( numberRows, plotsPerRow,
                           figsize=(6 * plotsPerRow, numberRows * 1.5), dpi=90,
                           squeeze=False )
    fig.suptitle( 'Fourier spectra for first simulated snapshot', y=1, fontsize=14 )
    fig.tight_layout( pad=4, h_pad=2, w_pad=2 )

    binFrequencies = fft.BinFrequencies
    binFrequencies = binFrequencies[ binFrequencies < 250.0 ]
    spectra = fft.readSnapshot()

    for iChannel in range( fft.NumberChannels ):
        axRow = int(iChannel / plotsPerRow)
        axCol = int(iChannel - axRow * plotsPerRow)
        channelPower = np.absolute( spectra[0:len(binFrequencies),iChannel] ) / fft.FftLength
        _ = ax[axRow, axCol].plot( binFrequencies, channelPower )
        _ = ax[axRow, axCol].set_title( 'Element %d' % (iChannel + 1) )
        _ = ax[axRow, axCol].grid( True, which='both' )

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