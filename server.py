import flwr as fl 

strategy = fl.server.strategy.FedAvg()

#Start flower
fl.server.start_server(
    server_address= "0.0.0.0:8080",
    config = fl.server.ServerConfig(num_rounds=1,round_timeout=None),
    strategy = strategy
)