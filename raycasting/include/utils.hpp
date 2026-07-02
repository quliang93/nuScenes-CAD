//
//
//

#pragma once

#include <filesystem>
#include <algorithm>

using namespace std;

void listdir(const string &dir, vector<filesystem::path> &output)
{
    vector<filesystem::path> files;
    copy(filesystem::directory_iterator(dir), filesystem::directory_iterator(), back_inserter(files));
    sort(begin(files), end(files));
    for (const auto &file: files)
        output.emplace_back(file);
}

string zfill(const string &src, int len)
{
    stringstream ss;
    ss << setfill('0') << setw(len) << src;
    return ss.str();
}
