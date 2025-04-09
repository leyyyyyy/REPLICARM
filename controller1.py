import pyfirmata

comport = 'COM4'
board = pyfirmata.Arduino(comport)
servo = board.get_pin('d:11:s')  # Servo connected to digital pin 9


def set_servo_angle(angle):
    servo.write(angle)


def cleanup():
    board.exit()