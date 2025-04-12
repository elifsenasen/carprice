namespace CarPriceWebApi.Models
{
    public class CarInput
    {
        public int Year { get; set; }
        public float Present_price { get; set; }
        public int Kms_driven { get; set; }
        public string Fuel_Type { get; set; }
        public string Seller_Type { get; set; }
        public string Transmission { get; set; }
        public int Owner { get; set; }
    }
}
