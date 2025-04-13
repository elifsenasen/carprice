using Microsoft.AspNetCore.Mvc;
using System.Net.Http;       
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using CarPriceWebApi.Models; // Model klasöründen CarInput sınıfı

namespace CarPriceWebApi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CarPriceController : ControllerBase
    {
        private readonly HttpClient _httpClient;

        public CarPriceController(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient();
        }

        public class PredictionResponse
        {
            public float Predicted_Price { get; set; }
        }

        [HttpPost("predict")]
        public async Task<IActionResult> Predict([FromBody] Car features)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = null // BU satır çok önemli!
            };

            var response = await _httpClient.PostAsJsonAsync("http://localhost:8000/predict", features, options);

            if (response.IsSuccessStatusCode)
            {
                var prediction = await response.Content.ReadFromJsonAsync<PredictionResponse>();
                return Ok(prediction);
            }

            return StatusCode((int)response.StatusCode, "Tahmin alınırken hata oluştu.");
        }
    }
    
}