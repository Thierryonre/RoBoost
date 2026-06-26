import socket
from json import dumps, loads
from threading import Thread
from time import sleep

class Network:
	def __init__(
		self,
		receivingPort=5556,
	):
		# Setup the sockets
		self.receivingSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.receivingPort = receivingPort
		self.receivingSock.bind(("0.0.0.0", receivingPort))

		# Store the latest data received
		self.latestData = {}

		# Start the receiver in a background thread so the main program isn't blocked
		receiverThread = Thread(target=self.updateLoop, daemon=True).start()

		# Socket for sending to arm
		print(f"Network: Prepared to send and receive data")

	def updateLoop(self):
		while True:
			try:
				data, _ = self.receivingSock.recvfrom(4096)
				print("Network: Received data")
				message = loads(data.decode("utf-8"))
				self.latestData = message
			except Exception as e:
				print(f"Network: Error in update loop: {e}")
				break

			sleep(0.4)

	def close(self):
		self.receivingSock.close()

		print("Network: Socket closed")

if __name__ == "__main__":
	net = Network()

	try:
		while True:
			print(f"Network: Latest data received: {net.latestData}")
			sleep(0.5)
	except KeyboardInterrupt:
		net.close()
