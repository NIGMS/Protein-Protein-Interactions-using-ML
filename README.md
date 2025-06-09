<p align="center">
    <img src="images/diagram.png" width="400" />
</p>

-------------

## **Contents**

- [Background](#background)
- [Overview](#overview)
- [Before Starting](#before-starting)
- [Getting Started](#getting-started)
- [Software Requirements](#software-requirements)
- [Architecture Design](#architecture-design)
- [Data](#data)
- [Funding](#funding)
- [License for Data](#license-for-data)

## **Background**
In this module, you wil learn how to apply machine learning (ML) methods to identify critical protein-protein interactions. This will allow you to distinguish the critical residues in making SARS-CoV-2 so much more infectious and lethal compared to SARS-CoV. 

The set of data that we have provided is molecular dynamics trajectories. Molecular dynamics (MD) is a type of biophysical simulation that looks how every atom in a particular biological system interacts with each other over a certain timescale. The output is known as a trajectory. It can be thought of as the "computational microscope into the cell." However, MD trajectories are a significant amount of data and can be difficult to analyze and parse through visually. 

This is where machine learning comes into play. We can take our MD trajectories and use that as an input for machine learning methods to analyze for us. In this particular situation, we have run MD on two distinct systems and would like to know the differences between them. More specifically, the two systems are SARS-CoV bound to the human receptor, hACE2, versus SARS-CoV-2 bound to hACE2. We would like to know exactly which residues on SARS-CoV-2 ended up driving the increased infectivity that led to the global pandemic we experiences in 2020. 



## **Overview**
In this tutorial, we will apply three different ML approaches - random forest, logistic regression, and multi-layer perceptron. We will analyze the residues at the interface of where SARS-CoV or SARS-CoV2 bind to hACE2. More specifically, the input data is the inverse of the distance between the residues. The tutorial is split into three separate modules. <br><br>
<span> <b>:sparkles: Click the image below to watch overview video</b> </span>
<p align="center">
        <a href="https://www.youtube.com/watch?v=FVe2cXwiGK0&list=PLXaEJPtnQ4w7Vu7vqWbttBjUGrPp4Qa7b&index=51">
            <img src="images/overview-video-graphic.jpeg" alt="MeRIP-Seq Module Overview Video", width="400"/>
        </a>
    <br><br>
</p>

## **Module Design**

### Submodule 1 - Introduction To Pymol
This section introduces PyMOL, a powerful tool for visualizing molecular structures.

You’ll learn how to load protein models, navigate the 3D interface, highlight important features, and create clear visualizations. Molecular visualization is key for understanding how proteins function, how they interact with other molecules, and how structural changes can impact biology.

By the end of this tutorial, you’ll be able to explore molecular structures confidently and prepare them for further analysis.

### Submodule 2 - Introduction To Machine Learning
This section provides a general introduction to machine learning, covering essential concepts and common models.

You’ll learn about key techniques such as random forests, logistic regression, and neural networks, and how these models are trained to find patterns and make predictions from data. The focus is on building an intuitive understanding of how machine learning works, laying the groundwork for applying these methods to more complex problems later.

By the end of this tutorial, you’ll be comfortable with basic machine learning principles and ready to explore real-world applications.

### Submodule 3 - Applying ML to SARS-CoV-2
In this section, you’ll apply machine learning to analyze data from molecular dynamics (MD) simulations—a kind of computational microscope that captures how molecules move and interact over time.

You’ll use three machine learning models—random forests, logistic regression, and neural networks—to uncover which residues and interactions are most critical for important molecular behaviors, like protein binding and structural shifts.

By the end of this tutorial, you’ll know how to combine dynamic molecular data with machine learning tools to reveal hidden patterns and drive new biological discoveries.

## Architecture Design
<center><img src="images/Architecture_Diagram.png" width=1500 /></center><br><br>

## **Before Starting**
This module is designed to run on the Google Cloud Platform (GCP). Follow the instructions below to prepare to run the module on GCP.

<p align="center">
        <a href="https://www.youtube.com/watch?v=Y27-IW6hj6E&list=PLXaEJPtnQ4w7Vu7vqWbttBjUGrPp4Qa7b&index=9">
            <img src="images/google_setup.png" alt="Creating an Vertex AI Notebook Instance", width="550"/>
        </a>
    <br>
    <span> Click above image to watch notebook setup video </span>
</p>

#### Setting up GCP

This module is designed to run on the Google Cloud Platform (GCP).
Follow the instructions below to prepare to run the module on GCP.

<details>

<summary>Setting up GCP</summary>

See the [Vertex AI Quickstart instructions](https://cloud.google.com/vertex-ai/docs/workbench/instances/create-console-quickstart) for details on steps 1-5.

1. Create a Google Cloud account
2. Create a Google Cloud project
3. Enable billing for your Google Cloud project
4. Go to Vertex AI Workbench and create a new instance using "CREATE NEW" -> "ADVANCED OPTIONS" and use the following configurations:
   * Details:<br/>&emsp;Region: us-east4<br/>&emsp;Zone: us-east4-a<br/>&emsp;Workbench type:<br/>&emsp;&emsp;Type: Instance
   * Environment:<br/>&emsp;JupyterLab Version: JupyterLab 4.x
   * Machine type:<br/>&emsp;Series: e2<br/>&emsp;Machine type: e2-standard-8<br/>&emsp;Idle shutdown:<br/>&emsp;&emsp;Enable Idle Shutdown: Checked<br/>&emsp;&emsp;Time of inactivity before shutdown (Minutes): 60
   * Disks: Use default settings
   * Networking:<br/>&emsp;Assign external IP address: Checked<br/>&emsp;Allow proxy access: Checked
   * IAM and security<br/>&emsp;Security options:<br/>&emsp;&emsp;Root access to the instance: Checked<br/>&emsp;&emsp;File downloading: Checked<br/>&emsp;&emsp;Terminal access: Checked
   * System health: Use default settings
5. Click "OPEN JUPYTERLAB" on your instance to open JupyterLab
6. To clone the Github repository for this module in JupyterLab, open a Terminal (File -> New Launcher -> Terminal) and run the following commands:
```bash
cd ~
git clone https://github.com/NIGMS/Protein-Protein-Interactions-using-ML.git
```
<br>    
After the last command completes there should be a folder name *Protein-Protein-Interactions-using-ML* that contains each submodule directory. Start with Submodule 0 to confirm you can spin up the PyMOL and AutoDock GUIs.    

</details>

## **Getting Started**
To begin go through the [pymol_notebook](submodule_0_PyMOL-GUI.ipynb) to start the Pymol instances. Once this is working go through sumbodules 1-3 in order. 

## **Software Requirements**
Access to PyMOL is provided through Submodule 0. All other required software is either downloaded through code execution within the notebook or, in limited situations, accessed through internal sites using the provided links.

## **Data**
The molecular dynamics data provided in this tutorial is from this publication originally.

Pavlova, A., Zhang, Z., Acharya, A., Lynch, D.L., Pang, Y.T., Mou, Z., Parks, J.M., Chipot, C. and Gumbart, J.C., 2021. Machine learning reveals the critical interactions for SARS-CoV-2 spike protein binding to ACE2. The Journal of Physical Chemistry Letters, 12(23), pp.5494-5502.

https://pubs.acs.org/doi/10.1021/acs.jpclett.1c01494

## **Funding**
This resource was supported with funds from NIH grant P20 GM103424-21 

## **License for Data**
Text and materials are licensed under a Creative Commons CC-BY-NC-SA license. The license allows you to copy, remix and redistribute any of our publicly available materials, under the condition that you attribute the work (details in the license) and do not make profits from it. More information is available here.




