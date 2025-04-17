import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(MaterialApp(home: CarPriceApp()));

class CarPriceApp extends StatefulWidget {
  const CarPriceApp({super.key});

  @override
  _CarPriceAppState createState() => _CarPriceAppState();
}

class _CarPriceAppState extends State<CarPriceApp> {
  final TextEditingController priceController = TextEditingController();
  final TextEditingController kmsController = TextEditingController();
  final TextEditingController yearController= TextEditingController();

  String? selectedFuelType;
  String? selectedSellerType;
  String? selectedTransmission;
  String? selectedOwner;
  String? selectedModel;
  String currentGif = "";
  Key gifKey = UniqueKey();

  bool _isLoading = false; // Loading durumunu tutacak değişken
  final AudioPlayer _audioPlayer = AudioPlayer();

  @override
    void initState() {
  super.initState();
  currentGif = "assets/gifs/initial.gif"; 
}

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
    child: Container(
      width: double.infinity, 
      child: DropdownButtonFormField<String>(
        value: selectedValue,
        onChanged: onChanged,
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(), 
        ),
        onTap: () => _updateGif(gifPath),
        items: items.map((String value) {
          return DropdownMenuItem<String>(value: value, child: Text(value));
        }).toList(),
      ),
    ),
  );
}
  void showPredictionOverlay(BuildContext context, double predictedPrice, double accuracy) {
    double minValue = 0;
    double maxValue = 100000;
    double percentage = (predictedPrice - minValue) / (maxValue - minValue);

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          title: Text("Predicted Price", textAlign: TextAlign.center),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                "\$${predictedPrice.toStringAsFixed(0)}",
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: Colors.green,
                ),
              ),
              SizedBox(height: 10),
              Text(
                "Accuracy: ${accuracy.toStringAsFixed(2)}%",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
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
                  Text("\$${minValue.toInt()}"),
                  Text("\$${(maxValue / 1000).toStringAsFixed(0)}K"),
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

  Future<void> _predictPrice() async {
    if (selectedModel == null) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text("Warning"),
      content: Text("Please select a model before predicting."),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text("OK"),
        ),
      ],
    ),
  );
  return;
}
    
    final String apiUrl = "http://localhost:5266/CarPrice/predict/$selectedModel";
    final String evaluateUrl= "http://localhost:5266/CarPrice/evaluate/$selectedModel";

    final carData = {
      'Year': int.tryParse(yearController.text),
      'Present_price': double.tryParse(priceController.text),
      'Kms_driven': int.tryParse(kmsController.text),
      'Fuel_Type': selectedFuelType,
      'Seller_Type': selectedSellerType,
      'Transmission': selectedTransmission,
      'Owner': selectedOwner != null ? int.parse(selectedOwner!) : null,
    };
    print("Car data being sent: ${jsonEncode(carData)}");

    try {
      setState(() {
        _isLoading = true;
      });

      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(carData),
      );
      final Map<String, dynamic> responseData = jsonDecode(response.body);

      if (responseData.containsKey('predicted_Price')) {
        final double predictedPrice = responseData['predicted_Price'] as double;
        final evalResponse = await http.post(Uri.parse(evaluateUrl));
        final evalData = jsonDecode(evalResponse.body);
        final double accuracy = evalData['results'] ?? 0.0;

        await _audioPlayer.play(
          AssetSource('sounds/success.mp3'),
        ); 

        showPredictionOverlay(context, predictedPrice, accuracy);
      } else {
        print("asdasd");
        // Handle error from the backend
        showDialog(
          context: context,
          builder:
              (context) => AlertDialog(
                title: Text("Error"),
                content: Text(
                  "Could not predict the car price. Please try again.",
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: Text("Ok"),
                  ),
                ],
              ),
        );
      }
    } catch (e) {
      print(e);

      showDialog(
        context: context,
        builder:
            (context) => AlertDialog(
              title: Text("Error"),
              content: Text("An error occurred. Please try again later."),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text("Ok"),
                ),
              ],
            ),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    List<String> years = List.generate(
      46,
      (index) => (1980 + index).toString(),
    );

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
              SizedBox(height: 15),
              buildInput(
                label: "Selected Model",
                selectedValue: selectedModel,
                items: ['lr', 'rf', 'xgb', 'svr', 'lgb'],
                onChanged:
                    (value) => setState(() {
                      selectedModel = value;
                    }),
                gifPath: "assets/gifs/model.gif",
              ),
              buildInput(
                label: "Fuel Type",
                selectedValue: selectedFuelType,
                items: ["Petrol", "Diesel"],
                onChanged:
                    (value) => setState(() {
                      selectedFuelType = value;
                    }),
                gifPath: "assets/gifs/fuel.gif",
              ),
              buildInput(
                label: "Seller Type",
                selectedValue: selectedSellerType,
                items: ["Dealer", "Individual"],
                onChanged:
                    (value) => setState(() {
                      selectedSellerType = value;
                    }),
                gifPath: "assets/gifs/seller.gif",
              ),
              buildInput(
                label: "Transmission Type",
                selectedValue: selectedTransmission,
                items: ["Manual", "Automatic"],
                onChanged:
                    (value) => setState(() {
                      selectedTransmission = value;
                    }),
                gifPath: "assets/gifs/transmission.gif",
              ),
              buildInput(
                label: "Number of Owners",
                selectedValue: selectedOwner,
                items: ["0", "1"],
                onChanged:
                    (value) => setState(() {
                      selectedOwner = value;
                    }),
                gifPath: "assets/gifs/owner.gif",
              ),
              Container(
                width: double.infinity,
                child: TextField(
                  controller: yearController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: "Manufacturing Year",
                    border: OutlineInputBorder(),
                  ),
                  onTap: () => _updateGif("assets/gifs/year.gif"),
                ),
              ),
              SizedBox(height: 15),
              Container(
                width: double.infinity,
                child: TextField(
                  controller: priceController,
                  keyboardType: TextInputType.numberWithOptions(decimal: true),
                  decoration: InputDecoration(
                    labelText: "Present Price (\$)",
                    border: OutlineInputBorder(),
                  ),
                  onTap: () => _updateGif("assets/gifs/price.gif"),
                ),
              ),
              SizedBox(height: 15),
              Container(
                width: double.infinity,
                child: TextField(
                  controller: kmsController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: "Kms Driven",
                    border: OutlineInputBorder(),
                  ),
                  onTap: () => _updateGif("assets/gifs/kms.gif"),
                ),
              ),

              SizedBox(height: 20),
              if (_isLoading)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20.0),
                  child: CircularProgressIndicator(),
                )
              else
                ElevatedButton(
                  onPressed: _predictPrice,
                  child: Text("Predict"),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
