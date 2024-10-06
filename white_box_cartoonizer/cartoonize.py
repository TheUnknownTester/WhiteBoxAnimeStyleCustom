import os
import cv2
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import uuid
import skvideo
import subprocess

tf.disable_eager_execution()
import network
import guided_filter
from tqdm import tqdm


class WB_Cartoonize:

    def __init__(self, model_path):
        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(
            self.input_photo, network_out, r=1, eps=5e-3
        )

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if "generator" in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        h, w = (h // 8) * 8, (w // 8) * 8
        image = image[:h, :w, :]
        return image

    def cartoonize(self, image):
        try:
            image = self.resize_crop(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = self.sess.run(
                self.final_out, feed_dict={self.input_photo: batch_image}
            )
            output = (np.squeeze(output) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            return output
        except:
            return None

    def process_video(self, fname, frame_rate):
        ## Capture video using opencv
        cap = cv2.VideoCapture(fname)

        target_size = (int(cap.get(3)), int(cap.get(4)))
        output_fname = os.path.abspath(
            "{}/{}-{}.mp4".format(
                fname.replace(os.path.basename(fname), ""),
                str(uuid.uuid4())[:7],
                os.path.basename(fname).split(".")[0],
            )
        )

        out = skvideo.io.FFmpegWriter(
            output_fname, inputdict={"-r": frame_rate}, outputdict={"-r": frame_rate}
        )

        while True:
            ret, frame = cap.read()

            if ret:

                frame = self.cartoonize(frame)

                if frame is None:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)

                out.writeFrame(frame)

            else:
                break
        cap.release()
        out.close()

        final_name = "{}final_{}".format(
            fname.replace(os.path.basename(fname), ""), os.path.basename(output_fname)
        )

        p = subprocess.Popen(
            [
                "ffmpeg",
                "-i",
                "{}".format(output_fname),
                "-pix_fmt",
                "yuv420p",
                final_name,
            ]
        )
        p.communicate()
        p.wait()

        os.system("rm " + output_fname)

        return final_name
