import tkinter as tk
import pickle
import signal
import sys
from typing import Dict, List
from PIL import Image, ImageTk

EPISODE_LABEL = "Episode: {}/{}"
FRAME_LABEL = "Frame: {}/{}"
FPS_LABEL = "Speed: {} fps"

# FILE_NAME = "training_data.log"
# FILE_NAME = "train-002.log"
#FILE_NAME = "log2.log"
FILE_NAME = "train_logA.log"

COUNT_EPISODES = True

class LogViewer:
    @property
    def frame(self):
        return self._frame or 0

    @frame.setter
    def frame(self, value):
        self._frame = value
        self._frame_label.set(FRAME_LABEL.format(self._frame,self.nb_frames))

    @property
    def current_episode(self):
        return self._current_episode

    @current_episode.setter
    def current_episode(self, value):
        self._current_episode = value
        self._episode_label.set(EPISODE_LABEL.format(self._current_episode,self.nb_episodes))

    @property
    def FPS(self):
        return self._FPS

    @FPS.setter
    def FPS(self, value):
        self._FPS = value
        self._fps_label.set(FPS_LABEL.format(self._FPS))

    # @property
    # def nb_episodes(self):
    #     return self._nb_episodes
    
    # @nb_episodes.setter
    # def nb_episodes(self, value):
    #     self._nb_episodes = value
    #     try:
    #         self._episode_label.set(EPISODE_LABEL.format(self._current_episode,self.nb_episodes))
    #     except AttributeError:
    #         pass

    def __init__(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.root = tk.Tk()
        self.init_vars()
        self.next_episode()

        self.setup_widgets()

        # Start streaming loop
        self.root.after(1, self.update_image)

        self.bind_keys()

        self.root.mainloop()
        self.shutdown(signum=0)

    def init_vars(self):
        self._episode_label = tk.StringVar()
        self._frame_label = tk.StringVar()
        self._fps_label = tk.StringVar()
        self.nb_episodes = -1
        self.nb_frames = 0
        self.FPS = 45
        self.frame = 0
        self.current_episode = 0
        self.FP = None

    def setup_widgets(self):
        self.root.title("Log Handler")
        self.root.geometry("300x300")  #initial window size

        # Observation streaming panel
        self.stream_panel = tk.Label(self.root,borderwidth=2, relief="groove")
        self.stream_panel.pack(side=tk.TOP, fill="both", expand="yes")#, fill="both", expand="yes")

        # Information frame
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.BOTTOM, fill="x", expand="yes")
        self.info_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(1, weight=1)


        # self.info_cf = tk.Label(self.info_frame, textvariable=self._frame_label, anchor=tk.W,borderwidth=2, relief="groove")
        # self.info_cf.pack(fill="both", expand="yes")

        self.info_ep = tk.Label(self.info_frame, textvariable=self._episode_label)
        self.info_ep.grid(row=0,column=0, sticky=tk.W)

        self.info_file = tk.Label(self.info_frame, text=f"File: '{FILE_NAME}'")
        self.info_file.grid(row=0,column=1, sticky=tk.W)

        self.info_cf = tk.Label(self.info_frame, textvariable=self._frame_label)
        self.info_cf.grid(row=1,column=0, sticky=tk.W)

        self.info_fps = tk.Label(self.info_frame, textvariable=self._fps_label)
        self.info_fps.grid(row=1,column=1, sticky=tk.W)


    def bind_keys(self):
        self.root.bind("<Return>", self.replay)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.shutdown(0))
        # self.root.bind('<Configure>', self.adjust_size)

    def load_data(self, data_file: str = "training_data.log") -> List:
        if self.FP is None:
            self.FP = open(FILE_NAME, "rb")
            if self.nb_episodes == -1:
                self.nb_episodes = self.count_episodes()

        self.data = pickle.load(self.FP)
        print("Pickled new data")
        self.nb_frames = len(self.data)

    def count_episodes(self):
        if not COUNT_EPISODES:
            return -1
        nb_episodes = 0
        print("Computing number of episodes ...")
        while True:
            try:
                pickle.load(self.FP)
                nb_episodes += 1
            except EOFError:
                print(f"Found {nb_episodes} episodes.")
                self.FP.seek(0)
                return nb_episodes

    def update_image(self):
        try:
            sample = self.data[self.frame]['step'][0]
        except IndexError:
            print("outofbound")
            self.next_episode()
            self.root.after(int(1000),lambda: self.update_image())
            return
        self.frame = self.frame+1
        img_array = Image.fromarray(sample)
        # if not (self.width, self.height) == (200, 150):
        #     # img_array = img_array.resize((self.width, self.height))
        #     print(f"Actual size: {self.stream_panel.winfo_width()}")

        self.stream_panel.update()
        img_array = img_array.resize(
            (self.stream_panel.winfo_width()-4,
            self.stream_panel.winfo_height()-4)
        )

        img =  ImageTk.PhotoImage(image=img_array)
        self.stream_panel.configure(image=img)
        self.stream_panel.image = img
        self.root.after(int(1000/self.FPS),lambda: self.update_image())

    def next_episode(self):
        self.current_episode += 1
        self.load_data()
        self.frame = 0

    def replay(self, event):
        self.frame = 0

    def commit(self, event):
        #TODO
        pass

    def next(self, event):
        #TODO
        pass

    def previous(self, event):
        #TODO
        pass

    def delete(self, event):
        #TODO
        pass

    def speedup(self, event):
        global FPS
        FPS += 5

    def speeddown(self, event):
        global FPS
        FPS -= 5

    def adjust_size(self, event):
        self.width = event.width-5
        self.height = event.height-5
        print(f"Set width to {event.width}")

    # TODO Label and info (+episode info)

    # def _resize_image(self, event):
    #     self.new_width = event.width
    #     self.new_height = event.height

    #     self.image = self.img_copy.resize((new_width, new_height))

    #     self.background_image = ImageTk.PhotoImage(self.image)
    #     self.background.configure(image =  self.background_image)

    def shutdown(self, signum, frame=None):
        print(f"Clean shutdown")
        try:
            FP.close()
        except NameError:
            pass
        sys.exit(signum)

if __name__=="__main__":
    LogViewer()
