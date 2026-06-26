import socket
from json import dumps, loads
from threading import Thread
from time import sleep

class Network:
	def __init__(
		self,
		transmittingAddress="192.168.11.5",
		receivingPort=5555,
		transmittingPort=5556,
	):
		# Setup the sockets
		self.transmittingAddress = transmittingAddress
		self.transmittingPort = transmittingPort
		self.receivingSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.transmittingSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.transmittingSock.setsockopt(socket.SOL_SOCKET, 25, "enp86s0".encode("utf-8"))
		self.receivingSock.bind(("localhost", receivingPort))

		# Start the receiver in a background thread so the main program isn't blocked
		receiverThread = Thread(target=self.updateLoop, daemon=True).start()

		# Socket for sending to arm
		print(f"Network: Prepared to send and receive data")

	def updateLoop(self):
		while True:
			try:
				data, _ = self.receivingSock.recvfrom(4096)
				message = loads(data.decode("utf-8"))

				self.sendDataToArm(message, "global")
			except Exception as e:
				print(f"Network: Error in update loop: {e}")
				break

			sleep(0.4)

	def sendDataToArm(self, data, source):
		assert source in ("hand", "global")

		data["source"] = source
		message = dumps(data).encode("utf-8")

		# print(f"Sending from source {source}")
		self.transmittingSock.sendto(message, (self.transmittingAddress, self.transmittingPort))

	def close(self):
		self.transmittingSock.close()

		print("Network: Socket closed")

if __name__ == "__main__":
	net = Network()

	sampleData = {}
	net.sendDataToArm(sampleData)
	net.close()
