import 'package:flutter/material.dart';

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
                items: ["1", "2", "3", "4","5","6","7","8","9","10"],
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
              ElevatedButton(
                onPressed: () {
                  print("Year: $selectedYear");
                  print("Price: ${priceController.text}");
                  print("Kms: ${kmsController.text}");
                  print("Owner: $selectedOwner");
                  print("Fuel: $selectedFuelType");
                  print("Seller: $selectedSellerType");
                  print("Transmission: $selectedTransmission");
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
