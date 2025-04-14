using Microsoft.AspNetCore.Mvc;
using System.Net.Http;       
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using CarPriceWebApi.Models;

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

        [HttpPost("predict/{model_name}")]
        public async Task<IActionResult> Predict([FromBody] Car features, [FromRoute] string model_name)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = null 
            };
            var url = $"http://localhost:8000/predict/{model_name}";
            var response = await _httpClient.PostAsJsonAsync(url, features, options);

            if (response.IsSuccessStatusCode)
            {
                var prediction = await response.Content.ReadFromJsonAsync<PredictionResponse>();
                return Ok(prediction);
            }
            return StatusCode((int)response.StatusCode, "Error has occured while predicting...");
        }

        [HttpPost("evaluate/{model_name}")]
        public async Task<IActionResult> Evaluate([FromRoute] string model_name){

            var url = $"http://localhost:8000/evaluate/{model_name}";
            var response = await _httpClient.PostAsync(url,null);

            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<EvaluationResult>();
                return Ok(result);
            }
            return StatusCode((int)response.StatusCode, "Error has occured while evaluating...");
        }

        [HttpGet("plot/outliers")]
        public async Task<IActionResult> OutlierPlot(){
            
            var url = "http://localhost:8000/plot/outliers";
            var response = await _httpClient.GetAsync(url);

            if (response.IsSuccessStatusCode)
            {
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return File(imageBytes, "image/png"); 
            }
            return StatusCode((int)response.StatusCode, "Could not load the plot...");
        }

        [HttpGet("plot/removed_outliers")]
        public async Task<IActionResult> RemovedOutlierPlot(){
            
            var url = "http://localhost:8000/plot/removed_outliers";
            var response = await _httpClient.GetAsync(url);

            if(response.IsSuccessStatusCode){
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return File(imageBytes, "image/png"); 
            }
            return StatusCode((int)response.StatusCode, "Could not load the plot...");
        }

        [HttpGet("plot/km")]
        public async Task<IActionResult> PlotKm(){
            
            var url = "http://localhost:8000/plot/km";
            var response = await _httpClient.GetAsync(url);

            if(response.IsSuccessStatusCode){
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return File(imageBytes, "image/png"); 
            }
            return StatusCode((int)response.StatusCode, "Could not load the plot...");
        }

        [HttpGet("plot/age")]
        public async Task<IActionResult> PlotAge(){
            
            var url = "http://localhost:8000/plot/age";
            var response = await _httpClient.GetAsync(url);

            if(response.IsSuccessStatusCode){
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return File(imageBytes, "image/png"); 
            }
            return StatusCode((int)response.StatusCode, "Could not load the plot...");
        }

        [HttpGet("plot/fuel")]
        public async Task<IActionResult> PlotFuel(){
            
            var url = "http://localhost:8000/plot/fuel";
            var response = await _httpClient.GetAsync(url);

            if(response.IsSuccessStatusCode){
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return File(imageBytes, "image/png"); 
            }
            return StatusCode((int)response.StatusCode, "Could not load the plot...");
        }
    }
    
}