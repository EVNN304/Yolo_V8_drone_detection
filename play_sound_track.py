import simpleaudio as sa
import multiprocessing as mp
import time



class Sound_track:
    def __init__(self, q_flag:mp.Queue, path_to_sound_file="ww.wav"):
        self.q_flag, self.path = q_flag, path_to_sound_file



    def run(self):
        process = mp.Process(target=self.play_track, args=(), daemon=True)
        process.start()


    def play_track(self):
        wave_obj = sa.WaveObject.from_wave_file(self.path)

        while True:
            if not self.q_flag.empty():
                self.q_flag.get()
                play_obj = wave_obj.play()
                time.sleep(1.0)
                play_obj.wait_done()

