Department of Software Engineering, Ankara Science University
CENG 384 - DSP
Instructor: Prof. Dr. Hakan Çağlar
Software Requirements Specification
FACIAL IMAGE WARPING, AGING, AND EXPRESSION TRANSFORMATION
Group Members
Nisa Nur AKLAN – 220204012
Ceyda YILDIZ – 220204011
Emir Ulaş ÖZDAĞ – 220204041
Nisanur İLHAN – 220204053
Berke AKÇAY – 220204058
Ahmet Talha Akyol - 230201006
Software Requirements Specification (SRS)
Facial Image Warping, Aging, and Expression Transformation
System
1. Introduction
1.1 Purpose ofthe System
The purpose of the Facial Image Warping, Aging, and Expression Transformation system
is to provide a software platform that enables users to analyze facial images and apply
various transformations using computer vision and digital image processing techniques.
The system allows users to upload a facial image, automatically detect key facial
landmarks, and modify facial expressions such as enhancing a smile, raising eyebrows,
widening lips, and slimming the face based on these points. In addition, the system can
perform aging and rejuvenation simulations using frequency-domain image processing
techniques.
Furthermore, the system applies Fourier transform to analyze the effects of these
transformations on the image’s frequency characteristics, and evaluates the quality of
the transformations using numerical metrics such as MSE, PSNR, and SSIM.
1.2 Scope of the System
The system operates within the domain of facial image processing and computer vision.
The system processes frontal facial images and allows users to apply a variety of
controlled facial transformations.
The main functionalities ofthe system include uploading facial images, detecting facial
landmarks, performing geometric facial warping, simulating aging and de-aging effects,
and analyzing the resulting images using frequency-based and quantitative evaluation
methods.
Users will be able to manipulate facial expressions such as smiling, eyebrow movement,
lip widening, and face slimming through geometric transformations based on facial
landmarks.
In addition to performing these transformations, the system will provide analytical tools
such as Fourier transform visualization and quantitative metrics. These tools allow users
to compare the original and transformed images and observe how facial modifications
influence image structure.
2. System Overview
2.1 Overall Functionality
The system acts as a comprehensive, end-to-end pipeline forfacial image modification
and analysis. It allows users to input standard frontal face images and apply highly
controlled geometric and frequency-based transformations. Rather than functioning
merely as a visual photo editor, the system computes and displays the mathematical
and structural impact of every alteration. Users can interactively adjust transformation
intensities, visually compare the original and processed images side-by-side, and
immediately review quantitative evaluation metrics and frequency spectrum analyses.
2.2 Major Components and Modules
The system is logically divided into several interconnected modules that handle the
lifecycle of the image processing task:
Image Input s Preprocessing Module: Manages the ingestion, format validation,
normalization, and necessary grayscale conversion of user-uploaded images to prepare
them for accurate analysis.
Facial Landmark Detection Module: Automatically identifies and extracts key facial
coordinates to serve as precise anchor points for structural modifications.
Expression Manipulation Module: Executes geometric warping (e.g., triangle-based
warping) based on landmark displacement to simulate expressions such as smile
enhancement, eyebrow raising, lip widening, and face slimming.
Aging s De-Aging Simulation Module: Manipulates the image in the frequency domain,
adjusting high and low-frequency components to add realistic wrinkles and texture for
aging, or to smooth the skin for de-aging.
Analysis s Evaluation Module: Computes the 2D Fourier Transform for spectral energy
analysis and calculates standardized image quality metrics such as Mean Squared Error
(MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) to quantify
the exact differences between the original and modified images.
User Interface (UI) Module: Provides the interactive dashboard where users adjust
transformation parameters, toggle landmark visibility, view comparisons, and export
structured analysis reports.
2.3 Main Objectives and Capabilities
The primary objective ofthe system is to provide a transparent, quantifiable, and
reproducible approach to facial image modification. Its core capabilities include:
Applying precise, non-destructive geometric warps to specific facialregions while
maintaining natural proportions and preventing visual artifacts.
Simulating age progression and regression through advanced frequency manipulation
rather than standard pixel painting.
Generating objective, numerical evaluation reports that measure perceptual similarity,
structural integrity, and image quality shifts resulting from the applied transformations.
2.4 Operating System
In accordance with the project's hardware constraints, the system is designed to
operate on standard personal computers without requiring specialized graphical
processing units (GPUs) or proprietary hardware. It is intended to be a cross-platform
web and backend based tool, fully compatible with major modern operating systems,
including Windows, macOS, and standard Linux distributions.
3. Functional Requirements
3.1 Image Input and Preprocessing Module
FR-3.1.1
The system shall allow users to upload frontal face images in JPG or PNG format.
FR-3.1.2
The system shall validate the file format and resolution ofthe uploaded image before
processing.
FR-3.1.3
The system shall display the uploaded image to the user before any processing is
applied.
FR-3.1.4
The system shall automatically detect the facial region within the uploaded image.
FR-3.1.5
The system shall crop and normalize the detected facial region.
FR-3.1.6
The system shall convertthe image to grayscale when frequency analysis is required.
FR-3.1.7
The system shall resize images to a standardized resolution.
3.2 Facial Landmark Detection Module
FR-3.2.1
The system shall detect facial landmarks on the uploaded face image.
FR-3.2.2
The system shall use a pre-trained model for facial landmark detection.
FR-3.2.3
The system shall display the detected facial landmarks on the image.
FR-3.2.4
The system shall allow users to toggle the visualization offacial landmarks on or off.
FR-3.2.5
The system shall allow exporting the coordinates of facial landmarks.
3.3 Expression Manipulation Module
FR-3.3.1
The system shall create a smile enhancement effect by modifying the landmark points
around the mouth region.
FR-3.3.2
The system shall create an eyebrow raising effect by modifying the eyebrow landmark
points.
FR-3.3.3
The system shall create a lip widening effect by modifying the lip landmark points.
FR-3.3.4
The system shall create a face slimming effect by modifying the cheek and jaw landmark
points.
FR-3.3.5
The system shall apply geometric image deformation based on the displacement of
facial landmarks.
FR-3.3.6
The system shall use triangle-based deformation techniques.
FR-3.3.7
The system shall apply pixel interpolation.
FR-3.3.8
The system shall minimize visual distortions.
FR-3.3.G
The system shall allow users to control the intensity level of facial expression
transformations (e.g., smile enhancement, eyebrow raising, lip widening, and face
slimming).
3.4 Aging and De-Aging Simulation Module
FR-3.4.1
The system shall perform aging simulation using frequency-domain manipulation.
FR-3.4.2
The system shall enhance high-frequency components to create wrinkle effects.
FR-3.4.3
The system shall use filtering techniques.
FR-3.4.4
The system shall reduce high-frequency components for de-aging simulation.
FR-3.4.5
The system shall apply low-pass filters to smooth skin regions.
FR-3.4.6
The system shall preserve important facial edges.
FR-3.4.7
The system shall apply contrast and tone adjustments to enhance the visual realism of
aging and de-aging simulations.
FR-3.4.8
The system may integrate a GAN-based or diffusion-based model.
FR-3.4.G
The system may allow comparison between frequency-based and AI-based aging
results.
3.5 Frequency Domain Analysis Module
FR-3.5.1
The system shall compute the two-dimensional Fourier Transform.
FR-3.5.2
The system shall generate the frequency magnitude spectrum.
FR-3.5.3
The system shall compute the total spectral energy.
FR-3.5.4
The system shall compute high-frequency and low-frequency energy components.
FR-3.5.5
The system shall compute frequency energy ratios.
FR-3.5.6
The system shall display comparative frequency analyses.
3.6 Quantitative Evaluation Module
FR-3.6.1
The system shall compute Mean Squared Error (MSE).
FR-3.6.2
The system shall compute Peak Signal-to-Noise Ratio (PSNR).
FR-3.6.3
The system shall compute Structural Similarity Index (SSIM).
FR-3.6.4
The system shall display the results in tabular form.
FR-3.6.5
The system shall allow the results to be exported as CSV or PDF.
3.7 Visualization and User Interface Module
FR-3.7.1
The system shall display the original and transformed images side by side.
FR-3.7.2
The system shall allow users to adjusttransformation parameters.
FR-3.7.3
The system shall allow users to save processed images.
3.8 AI-Supported Functional Extensions
FR-3.8.1
The system may use deep learning models to transfer expressions from a reference
image.
FR-3.8.2
The system may estimate apparent age before and aftertransformation.
FR-3.8.3
The system may supportreal-time webcam-based facial manipulation.
4. Non-Functional Requirements
4.1 Performance Requirements
NFR-4.1.1
The system shall process a standard uploaded facial image within 5 seconds under
normal operating conditions.
NFR-4.1.2
The system shall perform facial landmark detection within 5 seconds for a standard
frontal face image.
NFR-4.1.3
The system shall generate frequency-domain analysis results within 5 seconds after an
image transformation is applied.
4.2 Reliability Requirements
NFR-4.2.1
The system shall detect and rejectinvalid or unsupported image formats without
causing system failure.
NFR-4.2.2
The system shall handle image processing errors gracefully and display appropriate
error messages to the user.
NFR-4.2.3
The system shall produce consistent and reproducible results when the same input
image and transformation parameters are used.
4.3 Usability Requirements
NFR-4.3.1
The system shall provide a graphical userinterface that allows users to perform all
operations withoutrequiring programming knowledge.
NFR-4.3.2
The system shall present transformation results and evaluation metrics in a clear and
readable format.
NFR-4.3.3
The system shall allow users to visually compare the original and transformed images
through an intuitive interface.
5. System Constraints
5.1 Budget Constraints
SC-5.1.1
The system shall be developed using open-source tools in order to reduce costs.
SC-5.1.2
The project shall avoid the use of paid software licenses whenever possible.
5.2 Hardware Constraints
SC-5.2.1
The system shall operate on standard personal computers without requiring specialized
hardware.
SC-5.2.2
The system shall be designed to run on computers with moderate processor and
memory capacity.