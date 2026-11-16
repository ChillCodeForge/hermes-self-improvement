FROM python:3.12-slim AS build
WORKDIR /src
COPY . .
RUN pip install --target /deps .

FROM gcr.io/distroless/python3-debian12
COPY --from=build /deps /deps
CMD ["/deps/bin/worker"]
