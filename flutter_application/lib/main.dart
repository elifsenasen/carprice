import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

void main() => runApp(MaterialApp(home: CarPriceApp()));

class CarPriceApp extends StatefulWidget {
  const CarPriceApp({super.key});

  @override
  _CarPriceAppState createState() => _CarPriceAppState();
}

class _CarPriceAppState extends State<CarPriceApp> {
  final TextEditingController priceController = TextEditingController();
  final TextEditingController kmsController = TextEditingController();

  String? selectedYear;
  String? selectedFuelType;
  String? selectedSellerType;
  String? selectedTransmission;
  String? selectedOwner;

  String currentGif = "";
  Key gifKey = UniqueKey();

  bool _isLoading = false; // Loading durumunu tutacak değişken
  final AudioPlayer _audioPlayer = AudioPlayer();

  void _updateGif(String path) {
    setState(() {
      currentGif = path;
      gifKey = UniqueKey();
    });
  }

  Widget buildInput({
    required String label,
    required String? selectedValue,
    required List<String> items,
    required Function(String?) onChanged,
    required String gifPath,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: DropdownButtonFormField<String>(
        value: selectedValue,
        onChanged: onChanged,
        decoration: InputDecoration(labelText: label),
        onTap: () => _updateGif(gifPath),
        items: items.map((String value) {
          return DropdownMenuItem<String>(
            value: value,
            child: Text(value),
          );
        }).toList(),
      ),
    );
  }

  void showPredictionOverlay(BuildContext context, double predictedPrice) {
    double minValue = 0;
    double maxValue = 1000000;
    double percentage = (predictedPrice - minValue) / (maxValue - minValue);

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          title: Text("Predicted Price", textAlign: TextAlign.center),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                "${predictedPrice.toStringAsFixed(0)} ₺",
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.green),
              ),
              SizedBox(height: 20),
              LinearProgressIndicator(
                value: percentage.clamp(0.0, 1.0),
                minHeight: 20,
                backgroundColor: Colors.grey[300],
                color: Colors.green,
              ),
              SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("${minValue.toInt()} ₺"),
                  Text("${maxValue ~/ 1000}K ₺"),
                ],
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text("Exit"),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    List<String> years = List.generate(46, (index) => (1980 + index).toString());

    return Scaffold(
      appBar: AppBar(title: Text("Car Price Predictor")),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              if (currentGif.isNotEmpty)
                Container(
                  padding: EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black),
                    borderRadius: BorderRadius.circular(8),
                    color: Colors.white,
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.asset(
                      currentGif,
                      key: gifKey,
                      width: 120,
                      height: 120,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),
              SizedBox(height: 12),
              buildInput(
                label: "Year of Manufacture",
                selectedValue: selectedYear,
                items: years,
                onChanged: (value) => setState(() {
                  selectedYear = value;
                }),
                gifPath: "assets/gifs/year.gif",
              ),
              buildInput(
                label: "Fuel Type",
                selectedValue: selectedFuelType,
                items: ["Petrol", "Diesel", "Electric", "Hybrid"],
                onChanged: (value) => setState(() {
                  selectedFuelType = value;
                }),
                gifPath: "assets/gifs/fuel.gif",
              ),
              buildInput(
                label: "Seller Type",
                selectedValue: selectedSellerType,
                items: ["Dealer", "Private"],
                onChanged: (value) => setState(() {
                  selectedSellerType = value;
                }),
                gifPath: "assets/gifs/seller.gif",
              ),
              buildInput(
                label: "Transmission Type",
                selectedValue: selectedTransmission,
                items: ["Manual", "Automatic"],
                onChanged: (value) => setState(() {
                  selectedTransmission = value;
                }),
                gifPath: "assets/gifs/transmission.gif",
              ),
              buildInput(
                label: "Number of Owners",
                selectedValue: selectedOwner,
                items: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                onChanged: (value) => setState(() {
                  selectedOwner = value;
                }),
                gifPath: "assets/gifs/owner.gif",
              ),
              SizedBox(height: 12),
              TextField(
                controller: priceController,
                keyboardType: TextInputType.numberWithOptions(decimal: true),
                decoration: InputDecoration(labelText: "Present Price (\$)"),
                onTap: () => _updateGif("assets/gifs/price.gif"),
              ),
              TextField(
                controller: kmsController,
                keyboardType: TextInputType.number,
                decoration: InputDecoration(labelText: "Kms Driven"),
                onTap: () => _updateGif("assets/gifs/kms.gif"),
              ),
              SizedBox(height: 20),

              // Eğer loading durumundaysa butonun yerine CircularProgressIndicator göster
              if (_isLoading)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20.0),
                  child: CircularProgressIndicator(),
                )
              else
                ElevatedButton(
                  onPressed: () async {
                    // Ses ve gif hemen gelir
                    await _audioPlayer.play(AssetSource('sounds/success.mp3'));

                    // Loading durumunu aç
                    setState(() {
                      _isLoading = true;
                    });

                    // 1 saniye bekleyelim, sonra loading'i kapatalım
                    await Future.delayed(Duration(seconds: 1));

                    // Loading durumu kapandı
                    setState(() {
                      _isLoading = false;
                    });

                    // Dummy predicted price
                    double predictedPrice = 450000.0;

                    // Prediction Overlay ekranını göster
                    showPredictionOverlay(context, predictedPrice);
                  },
                  child: Text("Predict"),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
