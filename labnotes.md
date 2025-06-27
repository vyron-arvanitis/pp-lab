<h1>About the B-Factory</h1>

**1. What is a B-Factory?**

A B-Factory is a type of electron-positron collider designed to produce large numbers of B-mesons. B-mesons are particles containing a bottom quark with a mean lifetime of $1.638 \cdot 10^{-12}$. The term "factory" reflects the high luminosity needed to generate millions of B-meson pairs, primarily to study CP violation and other rare processes in the B-quark sector.

**2. What advantages do e+/e- colliders have compared to pp colliders?**

$e^+/e^-$ collisions produce fewer background particles, making events easier to analyze. The initial momenta and quantum numbers are precisely known, unlike in $pp$ collisions where partons (quarks/gluons) carry variable fractions of the proton momentum. Also, in $e^+/e^-$ collisions, the full center-of-mass energy goes into the reaction, making the collider much more energy efficient.

**3. What particles does a B-meson typically decay into?**

B-mesons decay through weak interactions into a wide variety of final states. Common decay products include
* Charged leptons $e^-, \mu^-$ and $\nu^-$
* Kaons and pions $K^+, K^0, \pi^+, \pi^0$
* Charm mesons $D^0, D^+$
* Photons (radiative decays)

**4. Which particles can we actually see in the detector?**

Detectors can directly detect
* Charged particals, including electrons, muons, pions, kaons, and protons aswell as their antiparticles
* Photons
* Neutral hadrons  like $K^0_\text{L}$

Detectors can not see
* Neutrinos, which are inferred only from missing energy/momentum of the decay process
* Short-lived resonances like $D^0$ meson or charmonium ($J/\psi$)

<h1>Belle II online book</h1>

**Introduction**

A sloppy definition of background is "everything one does not want to analyze". The most basic way to separate the background from the signals are called "cuts". A cut is a selection over one quantity, that has some separation power between signal and background.

There are four different kinds of background one wants to consider

1. Everything you don't want to measure

    The most probable process is $e^+e^- \to e^+e^-$, which is can easily be cut out. 

2. Physics Background

    In a collision, multiple processes can occur. In particle physics, processes with higher probabilities can obscure rarer, more interesting decays. This unwanted interference is called the physics background. For example when studying the decay $B \to K^{(*)}l^+l^-$, the challange lies in distinguishing the signal from the more frequent $B \to J/\psi K^{(*)}$ processes, which have a much higher branching fraction.

3. Continuum Background
    
    A common background is fromed by non-resonant hadronic events $e^+e^- \to q\overline{q}$. Since these hadronic events produce many tracks per event, it is very likely to find randomly some combination of genuine tracks and clusters that mimic the wanted signal but aren't from a $B$ decay. This background can be suppressed to a certain extent, although many analyses leave some part of this background in the data sample as it is relatively straightforward to model and cutting too strictly on continuum suppression variables will hurt signal efficiency at some stage. 

4. Beam-induced background

    Beam-induced background are tracks and clusters that are not produced from the primary $e^+e^-$ colision, but from other interactions in the beam itself. 
---
**Data Taking**

The SuperKEKB moves electrons with 7 GeV/c and positrons with 4 GeV/c against each other. The center-of-mass energy is typically around 10.58 GeV, corresponding to the $\Upsilon(4S)$ resonance, which is a specific excited state of the bottomonium system, a bound state of a bottom quark ($b$) and its antiparticle ($\overline{b}$). The $\Upsilon(4S)$ decays almost exclusively into $B\overline{B}$ pairs (a $B^+B^-$ or $B^0\overline{B^0}$ pair).

The Belle II detector is build around the interaction region. Other than the wanted "On-Resonance" collision at $\sqrt{s} =$ 10.58 GeV, the Belle II can collect a bunch of other events:

* **Cosmic:** At the beginning and end of each run period Belle II acquires cosmic muons, which are used for performance studies and calibration. 

* **Beam:** Those are short data taking used to study the beam-induced background on the innter sub-detectors. The beams are taken by non colliding proccesses, to remove all the hard scattering events.

* **Scan:** Short data taking period performed at slightly different energeis. The goal is to measure the line shape of the $e^+e^-$ cross section .

* **Non-4S:** Other resonances than the $4S$, for example, the $\Upsilon(1S)$, or $\Upsilon(6S)$.

In order to capture only relevant events, the Belle II triggers and filters events. To do so, the data is run through the Belle II online system, which consists of Data Acquisition (DAQ), Level 1 Trigger (TRG) and the High Level Trigger (HLT). After running the data through this both sytsems, it reaches the first storage hard disk. 

---

**Reconstruction**

After obtaining the data of the measurement or simulation, we have events that correpsond to raw detector responses. The goal is to transform this into something more usable for analysis, in order to identify the original four-vectors of particles produced in the interaction.

* **Clustering**

    If a particle passes through a pixel detector, we expect a particle in one of the pixels. Since the particle can be registered by many pixels, it would be useful to cluster the information of multiple pixels in order to identify the particle. The measurment of the amount of ionisation per pixel can be used to calculate the weighted mean for obtaining the center position. Even mroe advanced algorithms can be used depending on the readout characteristics of the detector.

* **Tracking**

    Another way is to identify the trajectories of particles by finding patterns or clusters in the tracking detectors. 

* **Particle Identification**

    After identifying the tracks, one can determine the likelihood for the track belonging to different particle types. For the CDC we can calculate the total energy loss over the track length and compare this to the expected values for different particle types. 

---
**Analysis**

After processing all the data, we are now able to analyse the detection of the events. Since the file contains all the events, we need to reduce it to the small fraction we actually want to analyse. This process is called *skimming*. The goal is to produce smaller datasets, each amounting to few percents of the total dataset, that can be shared among several analyses.

The three informations that Belle II detectors can provide are about Momentum, Energy, and PID probability. In most cases only two of them are available for a given particle. Most particles can not survive long enough to be detected. For this reason one needs to reconstruct these particles by measuring their decay products. 






