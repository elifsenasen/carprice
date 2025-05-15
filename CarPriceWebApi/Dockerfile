FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build

WORKDIR /car-price

ADD *.csproj .

RUN dotnet restore

ADD . .

RUN dotnet publish -c Release -o out

FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime
WORKDIR /app

COPY --from=build /car-price/out .

ENTRYPOINT ["dotnet", "CarPriceWebApi.dll"]