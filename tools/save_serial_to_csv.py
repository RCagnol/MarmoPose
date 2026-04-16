import serial
import csv
import keyboard



arduino_port = "COM3"
baud = 9600
fileName="arduino_logging.csv"
pressed = False
ser = serial.Serial(arduino_port, baud)
headers = ["Time (ms)","Event"]

print("Connected to Arduino port:" + arduino_port)
with open(fileName, "w", newline="", buffering=1) as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()

    while not pressed:
        if ser.in_waiting > 0:
            getData=ser.readline()
            dataString = getData.decode('utf-8').strip()
            print(dataString)
            dataSplitted = dataString.split(" ")
            writer.writerow({'Time (ms)': dataSplitted[0], 'Event': ' '.join(dataSplitted[1:])})


        pressed = keyboard.is_pressed('esc')