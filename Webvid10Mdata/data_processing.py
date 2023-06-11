import os
import pandas as pd
import cv2

use_gpu = False  # set this to False if you want to use CPU instead

# define input and output directories
input_dir = "results_10M_val.csv"
output_dir = "your_dataset/videos_edge"

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# read csv file using pandas
df = pd.read_csv(input_dir)

# loop through each row in the csv file
for i, row in df.iterrows():
    # get the video ID, content URL, and name
    video_id = row['videoid']
    content_url = row['contentUrl']
    name = row['name']

    # set the input and output paths
    input_path = content_url
    output_path_mp4 = os.path.join(output_dir, f"{video_id}.mp4")
    output_path_txt = os.path.join(output_dir, f"{video_id}.txt")

    # check if the output video file already exists
    if os.path.exists(output_path_mp4):
        print(f"Skipping {video_id} as the video file already exists.")
        continue

    # read video using OpenCV
    cap = cv2.VideoCapture(input_path)

    # create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path_mp4, fourcc, fps, frame_size, isColor=False)

    # loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply Canny edge detection
            if use_gpu:
                # use GPU acceleration if available
                cuda = cv2.cuda.createCannyEdgeDetector()
                edges = cuda.detect(gray)
            else:
                # use CPU otherwise
                edges = cv2.Canny(gray, 100, 200)

            # write edge map to output video
            out.write(edges)

        else:
            break

    # release resources
    cap.release()
    out.release()

    # write name to output text file
    with open(output_path_txt, 'w', encoding='utf-8') as f:
        f.write(name)

    print(f"Processed {video_id} and saved the output files.")
