# cbir_system


<p>This project implements Image Search Engine through CBIR (content based image retrieval) approach with User feedback.
CBIR approach pays greater attention to global and local information, such as color, shape and texture.</p>
<p>The system show the user a sample of pictures and asks for rating from the user. Using these ratings, system re-queries and re-calcul the weight of each descriptor and repeats until the right images are found.</p>
<h2>Part 1. Feature Extraction</h2>
<p>Feature extraction is a means of extracting compact but semantically valuable information from images. This information is used as a signature for the image. Similar images should have similar signatures.</p>

<p>In this retrieval system, we implemented 3 main image features descriptors:</p>
<ul>
  <li>color-based</li>
      <p>hello</p>
  <p>HSV histogram</p>
  <p>To extract the color features from the content of an image, a proper color space(HSV in this case) and an effective color descriptor are determined through color histogram(Local color histogram).
</p>
  <li>texture-based</li>
  <li>shape-based</li>
</ul>



