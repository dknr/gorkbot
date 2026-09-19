.PHONY: build run vet clean test

BIN := gorkbot

build:
	go build -tags goolm -o $(BIN) .

run: build
	./$(BIN) -config config.json

vet:
	go vet -tags goolm ./...

test:
	go test -tags goolm ./...

clean:
	rm -f $(BIN)
