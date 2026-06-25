import socket
import json
import threading

class Network:
	def __init__(self, port=5555, host="0.0.0.0"):
		self.port = port
		self.host = host
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.bind((self.host, self.port))

		# Store the latest data received
		self.latest_data = {"valid": False, "count": 0, "targets": []}
		self._lock = threading.Lock()

		# Start the receiver in a background thread so the main program isn't blocked
		receiver_thread = threading.Thread(target=self.update_loop, daemon=True)
		receiver_thread.start()

		# print(f"Network: Listening for UDP data on {self.host}:{self.port}")

	def update_loop(self):
		"""
		Run this in a background thread to keep latest_data refreshed.
		"""
		while True:
			try:
				data, _ = self.sock.recvfrom(4096)
				message = json.loads(data.decode("utf-8"))
				with self._lock:
					self.latest_data = message
			except Exception as e:
				print(f"Network: Error in update loop: {e}")
				break

	def getItemsInGlobalView(self):
		"""
		Returns the latest JSON object received.
		"""
		with self._lock:
			return self.latest_data

	def close(self):
		self.sock.close()
		print("Network: Socket closed.")

# --- Example Usage ---
if __name__ == "__main__":
	net = Network(port=5555)

	try:
		while True:
			# Main robot logic can poll the latest data instantly
			data = net.getItemsInGlobalView()
			if data["valid"]:
				print(f"Global View: Found {data['count']} targets.")

			# Simulate robot task frequency (e.g., 10Hz)
			import time
			time.sleep(0.1)
	except KeyboardInterrupt:
		net.close()
