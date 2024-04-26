import os
from moviepy.editor import VideoFileClip

unsplit_videos_directory = 'video'
file_name = 'a.mp4'
input_video_path = os.path.join(unsplit_videos_directory, file_name)
input_video_path = input_video_path.replace('\\', '/')
print(input_video_path)
clip = VideoFileClip(input_video_path)

clip_length = clip.duration

current_duration = clip_length
print("current_duration = " + str(current_duration))


if current_duration >= 500:
    divide_into_count = 10
    single_duration = current_duration / divide_into_count

    while current_duration > single_duration:
        subclip = clip.subclip(current_duration - single_duration, current_duration)

        file_base_name, file_extension = os.path.splitext(file_name)
        for i in range(divide_into_count):
            #fnum = str(i).format(:03)
            divided_filename = os.path.join(unsplit_videos_directory, f"{file_base_name}_{i:03}{file_extension}")
            file_check = divided_filename

            print('fname = ' + divided_filename)
            subclip = clip.subclip(current_duration - 2 * single_duration, current_duration - single_duration)
            subclip.to_videofile(divided_filename, codec="libx264", audio_codec="aac")
            subclip.close()
            current_duration -= single_duration
            print('current_duration: ' + str(current_duration) )

        #subclip = clip.subclip(current_duration - 2 * single_duration, current_duration - single_duration)
        #subclip.to_videofile(second_half_filename, codec="libx264", audio_codec="aac")

        #subclip.close()


        #print(f"Video split and saved: {first_half_filename} and {second_half_filename}")
        clip.close()
        delete_flag = True
else:
    d2 = current_duration / 60
    print("duration = " + str(d2))
