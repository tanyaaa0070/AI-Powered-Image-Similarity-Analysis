

# Image Similarity Web Application

This project provides a web-based tool to compare two images and compute their similarity using advanced feature extraction and robust matching techniques. Key technologies include OpenCV, scikit-learn, Pillow, and Flask.

## Features

- **Web interface** for uploading and comparing images.
- **Multi-level feature extraction**: color (RGB, HSV, LAB), texture (Sobel, LBP), and shape (edge, contour).
- **Cosine similarity** metric for robust and interpretable similarity scores.
- Supports popular image file formats (PNG, JPG, JPEG, BMP, GIF).

## Tech Stack

- Python 3.x
- Flask (Web framework)
- OpenCV (Computer vision/image processing)
- NumPy (Numerical processing)
- scikit-learn (Cosine similarity)
- Pillow (Image I/O processing)
- Werkzeug (File handling)

## How It Works

1. User uploads two images through the web interface.
2. Images are preprocessed and converted for robust feature extraction.
3. Color, texture, and shape features are extracted using OpenCV.
4. Feature vectors are built and compared using scikit-learn’s cosine similarity.
5. The web app displays a similarity score to the user.

## Getting Started

### Installation

```bash
git clone <your_repo_url>
cd <project_folder>
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Access the web interface at [http://localhost:5000](http://localhost:5000).

## Project Structure

| File              | Purpose                                                   |
|-------------------|----------------------------------------------------------|
| app.py            | Flask app, API routes, web interface, request handling    |
| model_utils.py    | Image feature extraction and similarity computation logic |
| requirements.txt  | Python dependency list                                    |

## Example Usage

- Upload your images using the form on the home page.
- The app will display the similarity score and analysis.

## Customization

You can tune the feature extraction logic (e.g., add more features or change weights for each type) by editing `model_utils.py`.

## License

MIT

## Acknowledgements

- OpenCV documentation and community for image processing algorithms.
- scikit-learn for powerful and easy-to-use similarity metrics.

***

Feel free to modify or add more project-specific details as needed for your deployment scenario.
