from Visualize_the_Board.Board import window
from kivy.config import Config
from kivy.core.window import Window

class GUI():

    def setup(self):

        width_of_board = 820
        height_of_board = 800

        #Set the Hight and Width of the App
        Config.set('graphics', 'width', str(width_of_board))
        Config.set('graphics', 'height', str(height_of_board))

        Config.set('graphics', 'resizable', '0')
        Config.write()

        Window.borderless = True

        Config.set('input', 'mouse', 'mouse,disable_multitouch')
        window().run()
