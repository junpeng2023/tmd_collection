
# Traditional Networks

1.
Oliver Zeidler (Seit November 2022)
(Oliver forscht an der Sicherheit von 5G/ 6G Kernnetzen und von Network Slicing.)
with
oliver.zeidler@tum.de

```
MA:
//Possibilities of Localization Mechanisms in a 5G Lab Environment

Mobile networks have long provided mechanisms for localization. This capability has been improved with LTE and new features in 5G allow even better positioning.

While some positioning methods are hard to recreate in a lab environment (such as AoA), others are possible (e.g. E-CID). One goal is to identify which can be recreated on-site.

Additionally, not much is known about the prevalence of support for these localization mechanisms.

According to their documentation, the Amarisoft Callbox supports the NL1-Interface between an external LMF and the built-in AMF. This can be used for an early prototype.

//Minimum goals:

·         Implement LMF that is able to interact with Amarisoft Callbox over NL1

·         Evaluate which localization methods are suitable for lab-based testing

·         Evaluate the prevalence of advertised localization mechanisms in commercial UEs

·         Evaluate the implementation status of localization mechanisms in commercial UEs

·         Evaluate if results can be explained by OS, Baseband or other factors

·         Find and evaluate possible attacks on the UEs location privacy

 

//Extended goals:

·         Implement LPP into Open5GS with AmariRAN or Open5GS with OAI

·         Implement Demo into the 5GCube framework


```

2.
Cristian Bermudez Serna (in October 2021)
(Research
Software Defined Networking 
Programmable Data Planes
Machine Learning
Network Function Virtualization)



```
1--MA:
//Planning and Evaluation of Unbalanced and Balanced PON for Rural Areas


Kurzbeschreibung:
With the increasing need for broadband in rural areas and the shift towards fiber-based Passive Optical Networks (PON), this research will focus on the deployment strategies of Balanced PON (BPON) and Unbalanced PON (UPON). The aim is to enhance efficiency and minimize infrastructure costs associated with broadband deployment. Recognizing the high expense associated with fiber deployment, especially in rural areas where geographical and demographic factors pose significant challenges, this study will focus on a shift from the conventional BPON approach, characterized by uniform power splitting (e.g., 1:16 or 1:32 splitting), to a more adaptable UPON strategy. The UPON architecture facilitates variable power splitting ratios, enhancing network reach and optimizing the distribution of network resources—such as bandwidth and optical power—in rural areas, where the distances between Optical Network Units (ONUs) or customer premises are greater than those in urban settings.
Beschreibung

With the increasing need for broadband in rural areas and the shift towards fiber-based Passive Optical Networks (PON), this research will focus on the deployment strategies of Balanced PON (BPON) and Unbalanced PON (UPON). The aim is to enhance efficiency and minimize infrastructure costs associated with broadband deployment. Recognizing the high expense associated with fiber deployment, especially in rural areas where geographical and demographic factors pose significant challenges, this study will focus on a shift from the conventional BPON approach, characterized by uniform power splitting (e.g., 1:16 or 1:32 splitting), to a more adaptable UPON strategy. The UPON architecture facilitates variable power splitting ratios, enhancing network reach and optimizing the distribution of network resources—such as bandwidth and optical power—in rural areas, where the distances between Optical Network Units (ONUs) or customer premises are greater than those in urban settings.

The methodology includes gathering detailed rural area data from OpenStreetMap to simulate realistic network designs, including the strategic placement of Optical Line Terminals (OLTs), splitters, and Optical Network Units (ONUs). The research will compare the traditional BPON, both single and cascading splitting, against UPON in terms of fiber length utilization, network elements placement strategy, power distribution efficiency, and overall cost-effectiveness. By using Gabriel graphs to generate rural area networks and analyzing PON equipment parameters like transmitted power range, sensitivity, and fiber attenuation, this research will be dedicated to identifying the most efficient, unprotected, optical access network configurations for rural settings. The research will also consider the potential of merging BPON and UPON strategies, aiming to harness the combined benefits of both architectures for a more versatile and cost-effective rural broadband deployment.

This comparative analysis is expected to keen insights into the scalability of BPON and UPON solutions, guiding network operators toward more informed infrastructure development decisions in rural settings. By analyzing multiple Gabriel graphs to evaluate total fiber length, splitter requirements, and infrastructure cost, this research aims to derive comprehensive guidelines for efficient fiber deployment in rural areas. These guidelines will contribute to internet access deployment strategies and help narrow the digital divide, showcasing UPON as an affordable and practical solution for rural broadband. By aligning with initiatives like the Broadband Programs in Germany and Bavaria, this research underscores the global effort to extend high-quality internet connectivity to underserved areas. [1] This analysis will thus empower network operators with the knowledge to select the most appropriate PON configuration for rural deployments, ensuring efficient, reliable, and affordable broadband access.

References:


[1] "State aid: Commission approves German scheme for very high capacity broadband networks in Bavaria." The European Sting, 29 Nov. 2019,
https://ec.europa.eu/commission/presscorner/detail/en/ip_19_6630.

//Voraussetzungen
Knowledge of:

Python
Object Oriented Programming
GIT
Optical Networks
Linux

2--BA:
//Data plane performance measurements
Stichworte:
P4, SDN
Kurzbeschreibung:
This work consists on performing measurements for a given P4 code on different devices.
Beschreibung

Software-Defined Networking (SDN) is a network paradigm where control and data planes are decoupled. The control plane consists on a controller, which manages network functionality and can be deployed in one or multiple servers. The data plane consists on forwarding devices which are instructed by the controller on how to forward traffic.

P4 is a domain-specific programming language, which can be used to define the functionality of forwarding devices as virtual or hardware switches and SmartNICs.

This work consists on performing measurements for a given P4 code on different devices. For that, an small P4-enabled virtual network will be used to perform some measurments. Later, data will be also collected from hardware devices as switchs and SmartNICs. Measurement should be depicted in a GUI for its subsequent analysis.

//Voraussetzungen
Basic knowledge on the following:

Linux
Networking/SDN
Python/C
Web programming (GUI).
Please send your CV and transcript of records.



```







