#include <cstdint>
#include <iostream>
#include <cmath>
#include <string>

class fxp {
private:
    int64_t value;  // Changed to signed
    
public:
    // Static scale factor (number of fractional bits)
    static uint32_t SCALE;
    
    // Constructors
    fxp() : value(0) {}
    
    explicit fxp(int64_t raw_value) : value(raw_value) {}
    
    fxp(double f) {
        value = static_cast<int64_t>(f * (1LL << SCALE));
    }
    
    fxp(int i) {
        value = static_cast<int64_t>(i) << SCALE;
    }
    
    // Getters
    int64_t get_raw() const { return value; }
    
    // Conversion to double
    double to_double() const {
        return static_cast<double>(value) / (1LL << SCALE);
    }
    
    // Conversion to float
    float to_float() const {
        return static_cast<float>(value) / (1LL << SCALE);
    }
    
    // Conversion to int (truncates fractional part)
    int to_int() const {
        return static_cast<int>(value >> SCALE);
    }
    
    // Static helper to create from raw value
    static fxp from_raw(int64_t raw) {
        return fxp(raw);
    }
    
    // Static helper to set scale
    static void set_scale(uint32_t new_scale) {
        SCALE = new_scale;
    }
    
    // Static helper to get scale
    static uint32_t get_scale() {
        return SCALE;
    }
    
    // Addition
    fxp operator+(const fxp& other) const {
        return fxp(value + other.value);
    }
    
    fxp& operator+=(const fxp& other) {
        value += other.value;
        return *this;
    }
    
    // Subtraction
    fxp operator-(const fxp& other) const {
        return fxp(value - other.value);
    }
    
    fxp& operator-=(const fxp& other) {
        value -= other.value;
        return *this;
    }
    
    // Multiplication
    fxp operator*(const fxp& other) const {
        // Multiply and shift back to maintain scale
        __int128 result = static_cast<__int128>(value) * other.value;
        return fxp(static_cast<int64_t>(result >> SCALE));
    }
    
    fxp& operator*=(const fxp& other) {
        __int128 result = static_cast<__int128>(value) * other.value;
        value = static_cast<int64_t>(result >> SCALE);
        return *this;
    }
    
    // Division
    fxp operator/(const fxp& other) const {
        // Shift left before division to maintain scale
        __int128 dividend = static_cast<__int128>(value) << SCALE;
        return fxp(static_cast<int64_t>(dividend / other.value));
    }
    
    fxp& operator/=(const fxp& other) {
        __int128 dividend = static_cast<__int128>(value) << SCALE;
        value = static_cast<int64_t>(dividend / other.value);
        return *this;
    }
    
    // Unary minus
    fxp operator-() const {
        return fxp(-value);
    }
    
    // Comparison operators
    bool operator==(const fxp& other) const {
        return value == other.value;
    }
    
    bool operator!=(const fxp& other) const {
        return value != other.value;
    }
    
    bool operator<(const fxp& other) const {
        return value < other.value;
    }
    
    bool operator<=(const fxp& other) const {
        return value <= other.value;
    }
    
    bool operator>(const fxp& other) const {
        return value > other.value;
    }
    
    bool operator>=(const fxp& other) const {
        return value >= other.value;
    }
    
    // Truncation (remove fractional bits towards zero)
    fxp truncate() const {
        int64_t truncated = (value >> SCALE) << SCALE;
        return fxp(truncated);
    }
    
    // Floor (round towards negative infinity)
    fxp floor() const {
        if (value >= 0) {
            return fxp((value >> SCALE) << SCALE);
        } else {
            int64_t mask = (1LL << SCALE) - 1;
            if (value & mask) {
                return fxp(((value >> SCALE) - 1) << SCALE);
            }
            return fxp(value);
        }
    }
    
    // Ceiling (round towards positive infinity)
    fxp ceil() const {
        if (value >= 0) {
            int64_t mask = (1LL << SCALE) - 1;
            if (value & mask) {
                return fxp(((value >> SCALE) + 1) << SCALE);
            }
            return fxp(value);
        } else {
            return fxp((value >> SCALE) << SCALE);
        }
    }
    
    // Round to nearest integer
    fxp round() const {
        int64_t half = 1LL << (SCALE - 1);
        if (value >= 0) {
            return fxp(((value + half) >> SCALE) << SCALE);
        } else {
            return fxp(((value - half) >> SCALE) << SCALE);
        }
    }
    
    // Absolute value
    fxp abs() const {
        return fxp(value >= 0 ? value : -value);
    }
    
    // Square root (using Newton's method, only for non-negative)
    fxp sqrt() const {
        if (value <= 0) return fxp(0);
        
        int64_t x = value;
        int64_t y = (x + (1LL << SCALE)) >> 1;
        
        while (y < x) {
            x = y;
            __int128 temp = static_cast<__int128>(value) << SCALE;
            y = (x + static_cast<int64_t>(temp / x)) >> 1;
        }
        
        return fxp(x);
    }
    
    // Print function
    void print() const {
        std::cout << to_double();
    }
    
    // Print with label
    void print(const std::string& label) const {
        std::cout << label << ": " << to_double();
    }
    
    // Stream output operator
    friend std::ostream& operator<<(std::ostream& os, const fxp& f) {
        os << f.to_double();
        return os;
    }
    
    // Stream input operator (reads as double)
    friend std::istream& operator>>(std::istream& is, fxp& f) {
        double d;
        is >> d;
        f = fxp(d);
        return is;
    }
};

// // Initialize static member (default scale of 32 bits for fractional part)
// uint32_t fxp::SCALE = 32;

// // Example usage and testing
// int main() {
//     // Set scale (e.g., 16 bits for fractional part)
//     fxp::set_scale(16);
    
//     std::cout << "Fixed-point scale: " << fxp::get_scale() << " bits\n\n";
    
//     // Create fixed-point numbers (including negatives)
//     fxp a(3.14159);
//     fxp b(-2.71828);
//     fxp c(-10);
//     fxp d(5.5);
    
//     std::cout << "a = " << a << std::endl;
//     std::cout << "b = " << b << std::endl;
//     std::cout << "c = " << c << std::endl;
//     std::cout << "d = " << d << std::endl;
//     std::cout << std::endl;
    
//     // Addition with negatives
//     fxp sum = a + b;
//     std::cout << "a + b = " << sum << std::endl;
    
//     // Subtraction
//     fxp diff = a - b;
//     std::cout << "a - b = " << diff << std::endl;
    
//     // Multiplication with negatives
//     fxp prod = a * b;
//     std::cout << "a * b = " << prod << std::endl;
    
//     fxp prod2 = b * c;
//     std::cout << "b * c = " << prod2 << std::endl;
    
//     // Division with negatives
//     fxp quot = d / b;
//     std::cout << "d / b = " << quot << std::endl;
//     std::cout << std::endl;
    
//     // Unary minus
//     std::cout << "-a = " << -a << std::endl;
//     std::cout << "-b = " << -b << std::endl;
//     std::cout << std::endl;
    
//     // Comparison with negatives
//     std::cout << "a > b: " << (a > b) << std::endl;
//     std::cout << "b < c: " << (b < c) << std::endl;
//     std::cout << "c < b: " << (c < b) << std::endl;
//     std::cout << std::endl;
    
//     // Truncation and rounding with negatives
//     fxp e(-5.678);
//     std::cout << "e = " << e << std::endl;
//     std::cout << "e.truncate() = " << e.truncate() << std::endl;
//     std::cout << "e.floor() = " << e.floor() << std::endl;
//     std::cout << "e.ceil() = " << e.ceil() << std::endl;
//     std::cout << "e.round() = " << e.round() << std::endl;
//     std::cout << std::endl;
    
//     // Absolute value
//     std::cout << "abs(b) = " << b.abs() << std::endl;
//     std::cout << "abs(a) = " << a.abs() << std::endl;
//     std::cout << std::endl;
    
//     // Square root
//     fxp f(16.0);
//     std::cout << "sqrt(16) = " << f.sqrt() << std::endl;
    
//     // Raw value access
//     std::cout << "\nRaw value of a: " << a.get_raw() << std::endl;
//     std::cout << "Raw value of b: " << b.get_raw() << std::endl;
    
//     // Conversion
//     std::cout << "\na as int: " << a.to_int() << std::endl;
//     std::cout << "b as int: " << b.to_int() << std::endl;
//     std::cout << "c as double: " << c.to_double() << std::endl;
    
//     return 0;
// }