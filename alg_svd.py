from base import BaseComm

class AlgSVD(BaseComm) : 
    def __init__(self, vehIDs: list[str]) -> None:
        self.data = None  # Row information 
        self.a = None
        self.x = None
        self.Y = None 
        super().__init__(vehIDs)
    
    def send(self, src: str, dest: list[str]) -> None:
        return super().send(src, dest)
    
    def receive(self, msg, src, dest) -> None:
        msg  = self.data
        return super().receive(msg, src, dest)
    
    def process(self):
        