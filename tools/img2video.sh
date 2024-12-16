ffmpeg -framerate 30 -start_number 0 -i init_iter_1050/init_iter_%d.png -c:v libopenh264 -pix_fmt yuv420p output.mp4

ffmpeg -framerate 10 -i ./frame_4_iter_150_cam_2/frame_4_cam_0_iter_%d_render.png -c:v libx264 -pix_fmt yuv420p output_video.mp4

