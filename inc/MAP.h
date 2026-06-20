#pragma once
#include <iostream>
#include <stdexcept>   // <-- add

class Map 
{
public:
    size_t graph_size = 0;
    int height, width;

    Map(){};

    Map(int width, int height) : height(height), width(width){
        map_data = new size_t*[height];
        ids = new size_t*[height];
        for(int i = 0; i < height; i ++){
            map_data[i] = new size_t[width];
            ids[i] = new size_t[width];
        }
    }

    void setSize(int _width, int _height){
        height = _height;
        width = _width;
        map_data = new size_t*[height];
        ids = new size_t*[height];
        for(int i = 0; i < height; i ++){
            map_data[i] = new size_t[width];
            ids[i] = new size_t[width];
        }
    }

    void setVal(int x, int y, size_t val){
        map_data[x][y] = val;
    }

    size_t getVal(int x, int y){
        return map_data[x][y];
    }

    void setID(int x, int y, size_t id){
        ids[x][y] = id;
    }

    // ---------- replace your current getID with this guarded version ----------
    size_t getID(int x, int y) const {
        // NOTE: your code uses x as row (0..height-1) and y as col (0..width-1)
        if (x < 0 || x >= height || y < 0 || y >= width) {
            std::cerr << "[Map::getID] OOB (x=" << x << ", y=" << y
                      << ") for map width=" << width << " height=" << height << "\n";
            throw std::out_of_range("Map::getID out of bounds");
        }
        return ids[x][y];
    }
    // -------------------------------------------------------------------------

    ~Map(){
       for (int i = 0; i < height; i++) {
            delete[] map_data[i];
            delete[] ids[i];
        }
        delete[] map_data;
        delete[] ids; 
    }

private:
    size_t ** map_data;
    size_t ** ids;
};
