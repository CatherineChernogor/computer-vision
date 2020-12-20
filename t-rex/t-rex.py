import pyautogui
import msvcrt
import matplotlib.pyplot as plt

def start_game():
    pyautogui.click(x=1920+100, y=200) 
    pyautogui.press('space')

def jump():
    pyautogui.press('space')

def check():
    image = pyautogui.screenshot(region=(0,200, 0+300, 500))
    plt.figure()
    plt.imshow(image)
    return True




start_game()
while(True):

    if check():
       
        plt.show() 
        jump()