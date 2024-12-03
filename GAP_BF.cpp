#include <bits/stdc++.h>
using namespace std;

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

void input_matrix(vector<vector<long long>> &v, int n, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> v[i][j];
        }
    }
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, modular;
    long long upper_limit, copy, current, best = 0;
    bool valid;
    cin >> n >> m;
    vector<long long> t(m), limit(m);
    vector<vector<long long>> w(m, vector<long long>(n)), p(m, vector<long long>(n));
    for (int i = 0; i < m; i++)
    {
        cin >> t[i];
    }
    input_matrix(w, n, m);
    input_matrix(p, n, m);
    upper_limit = pow(m, n);
    for (long long num = 0; num < upper_limit; num++)
    {
        copy = num;
        current = 0;
        valid = true;
        for (int i = 0; i < m; i++)
        {
            limit[i] = t[i];
        }
        for (int j = 0; j < n; j++)
        {
            modular = copy % m;
            limit[modular] -= w[modular][j];
            if (limit[modular] < 0)
            {
                valid = false;
                break;
            }
            current += p[modular][j];
            copy /= m;
        }
        if (valid)
        {
            best = max(best, current);
        }
    }
    cout << best;
    return 0;
}
