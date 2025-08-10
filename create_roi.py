from util_module.map_module import Map_ROI, Camera_ROI
from util_module.camera_module import VideoFileCamera

##################################################################
## map set up

# image_path = r"ref_image\map\book_fair_2019.jpg"
# try:
#     my_map = Map_ROI(image_path, scale_factor=1)
    
#     # Start in create mode
#     my_map.create_roi()
    
#     # Now, switch to edit mode (after saving and re-loading)
#     my_map.edit_roi()

# except FileNotFoundError as e:
#     print(e)
# finally:
#     pass


##################################################################
## 1st cam set up

video_path = r"Vid_test\front_room.mp4"  # Replace with your video file path
camera_instance = VideoFileCamera(video_path=video_path, scale_factor=1.5)
try:
    my_map = Camera_ROI(camera_instance)
    
    # Start in create mode
    my_map.create_roi()
    
    # Now, switch to edit mode (after saving and re-loading)
    my_map.edit_roi()

except FileNotFoundError as e:
    print(e)
finally:
    camera_instance.stop()

