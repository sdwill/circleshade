import numpy as np
cimport numpy as np
cimport libc.math as cmath
import matplotlib.pyplot as plt


cdef double segment_area_displacement(double radius, double displacement):
    """
    Compute the area of a circular segment, defined to be area between the edge of a circle and a
    chord through the circle, which intersects the edge at exactly two points.  The chord can
    be specified either by a distance from the center (d), or by the angle it subtends.

    Parameters
    ----------
    d : float
        Distance from the center of the circle
    theta : float
        Angle subtended by the circular sector formed by the points of intersection between
        the edge of the circle and the chord.

    Returns
    -------
    float
        Area of the cap.  Cannot be larger than the area of the circle itself.

    """
    return (radius * radius * cmath.acos(displacement / radius)
            - displacement * cmath.sqrt(radius * radius - displacement * displacement))


cdef double segment_area_theta(double radius, double theta):
    return 0.5 * radius * radius * (theta - cmath.sin(theta))


cdef double quadrant_area(double radius, double dh, double dv, long quadrant):
    """
    Compute the area of one of the four regions formed by the intersection of two orthogonal
    chords.  One of the chords is oriented vertically, while the other is horizontal.  The
    chords are specified by their distances from the circle center.  The areas are computed as
    follows:
        1) Draw a line connecting the points where the vertical and horizontal chords
        intersect the edge of the circle, respectively.  Since each chord intersects the
        circle at two locations (for a total of four intersection points), the choice of
        quadrant determines which of the two vertical intersection points and which of the
        two horizontal intersection points is chosen.

        2) Compute the area of the triangular portion of the quadrant (between the circle
        center and the line).

        3) Compute the area of the circular segment portion of the quadrant (between the line
        and the circle edge).

    The quadrant area is then the sum of the triangular part and segment part.

    Parameters
    ----------
    dh : float
        The displacement of the horizontal chord from the circle center.
    dv : float
        The displacement of the vertical chord from the circle center.
    quadrant : long
        The desired quadrant.  Quadrants are labeled 1, 2, 3, or 4, with 1 being the top-left
        quadrant and increasing clockwise around the circle.

    Returns
    -------
    float
        The area of the desired quadrant.

    """
    cdef double ell_h, ell_v
    cdef double left_segment, right_segment, top_segment, bottom_segment
    cdef double base, height, triangle_area, cap_line, theta, total

    # Return zero if the chords intersect outside the circle
    if cmath.sqrt(dh * dh + dv * dv) >= radius:
        return 0
    else:
        # Length of horizontal chord
        ell_h = 2 * cmath.sqrt(radius * radius - dh * dh)

        # Length of vertical chord
        ell_v = 2 * cmath.sqrt(radius * radius - dv * dv)

        # Compute the lengths of the four line segments formed by the intersection of the two
        # chords
        left_segment = dv + 0.5 * ell_h
        right_segment = 0.5 * ell_h - dv
        top_segment = 0.5 * ell_v - dh
        bottom_segment = dh + 0.5 * ell_v

        if quadrant == 1:
            base = left_segment
            height = top_segment
        elif quadrant == 2:
            base = right_segment
            height = top_segment
        elif quadrant == 3:
            base = right_segment
            height = bottom_segment
        elif quadrant == 4:
            base = left_segment
            height = bottom_segment

        triangle_area = 0.5 * base * height  # Area of triangular portion

        # Length of hypotenuse of triangular portion
        cap_line = cmath.sqrt(base * base + height * height)

        # Angle subtended by circular segment portion
        theta = 2 * cmath.asin(cap_line / (2 * radius))

        # Area of circular segment
        cap_area = segment_area_theta(radius, theta=theta)
        total = triangle_area + cap_area
        return total


def find_edges(double radius, (double, double) center, double[:] axis):
    cdef double xc = center[0]
    cdef double yc = center[1]
    cdef double step = axis[1] - axis[0]
    # cdef Py_ssize_t N = axis.shape[0]
    cdef Py_ssize_t N = axis.shape[0]
    cdef Py_ssize_t index, m_plus, m_minus, n_plus, n_minus
    cdef double displacement, discriminant, intersection
    cdef double edge_start = axis[0] - 0.5 * step
    cdef double [:] edge_lines = np.arange(edge_start, edge_start + (N + 1) * step, step,
                                           dtype=np.float64)
    # The axis contains the center locations of each pixel.  To find the edges, shift center by
    # a half pixel in one direction, and add the remaining edge in the other direction to the
    # end:
    #                   left edge ->  | * | * | * |  <- right edge
    # edge_lines = np.append(axis - 0.5 * step, [axis[-1] + 0.5 * step])
    edge_pixels = np.zeros((N, N), dtype=long)
    interior = np.zeros((N, N), dtype=long)
    arr = np.zeros((N, N), dtype=np.float64)

    cdef double [:] edge_lines_v = edge_lines
    cdef long [:, :] edge_pixels_v = edge_pixels
    cdef long [:, :] interior_v = interior
    cdef double[:, :] arr_v = arr
    cdef double radius_squared = radius ** 2  # Precompute this

    for index in range(0, N + 1):
        edge_line = edge_lines_v[index]

        # Vertical lines
        displacement = edge_line - xc  # Translate line, not circle
        discriminant = radius_squared - (displacement ** 2)

        # If discriminant is negative, line and circle do not intersect
        if discriminant >= 0:
            intersection = cmath.sqrt(discriminant)
            m_plus = <long>cmath.ceil((intersection - (edge_lines_v[0] - yc)) / step)
            edge_pixels_v[m_plus - 1, index - 1] = edge_pixels_v[m_plus - 1, index - 1] + 1
            edge_pixels_v[m_plus - 1, index] = edge_pixels_v[m_plus - 1, index] + 1

            # If discriminant is 0, then the line is tangent to the circle and the two
            # intersection points are the same, so we don't need to calculate the second one
            if discriminant > 0:
                m_minus = <long>cmath.ceil((-intersection - (edge_lines_v[0] - yc)) / step)
                edge_pixels_v[m_minus - 1, index - 1] = edge_pixels_v[m_minus - 1, index - 1] + 1
                edge_pixels_v[m_minus - 1, index] = edge_pixels_v[m_minus - 1, index] + 1

                for m in range(m_minus, m_plus):
                    interior_v[m, index - 1] = interior_v[m, index - 1] + 1
                    interior_v[m, index] = interior_v[m, index] + 1

                if m_minus == m_plus:
                    if displacement < 0:
                        col = index - 1   # Left of center, so column to left of circle has cap
                    else:
                        col = index  # Right of center, so column to right of circle has cap

                    arr_v[m_minus - 1, col] = segment_area_displacement(radius, abs(displacement))
                    edge_pixels_v[m_minus - 1, col] = -1

        # Horizontal lines
        displacement = edge_line - yc
        discriminant = radius_squared - (displacement ** 2)

        if discriminant >= 0:
            intersection = cmath.sqrt(discriminant)
            n_plus = <long>cmath.ceil((intersection - (edge_lines_v[0] - xc)) / step)
            edge_pixels_v[index - 1, n_plus - 1] = edge_pixels_v[index - 1, n_plus - 1] + 1
            edge_pixels_v[index, n_plus - 1] = edge_pixels_v[index, n_plus - 1] + 1

            if discriminant > 0:
                n_minus = <long>cmath.ceil((-intersection - (edge_lines_v[0] - xc)) / step)
                edge_pixels_v[index - 1, n_minus - 1] = edge_pixels_v[index - 1, n_minus - 1] + 1
                edge_pixels_v[index, n_minus - 1] = edge_pixels_v[index, n_minus - 1] + 1

                for n in range(n_minus, n_plus):
                    interior_v[index - 1, n] = interior_v[index - 1, n] + 1
                    interior_v[index, n] = interior_v[index, n] + 1

                if n_minus == n_plus:

                    if displacement < 0:
                        row = index - 1   # Below center, so row below circle has the cap
                    else:
                        row = index  # Above center, so row above circle has cap

                    arr_v[row, n_minus - 1] = segment_area_displacement(radius, abs(displacement))
                    edge_pixels_v[row, n_minus - 1] = -1

    # Any pixel with only 1 intersection must be the pixel whose interior edge was tangent
    # to the circle, and thus is not an edge pixel.  The pixel whose exterior edge was
    # tangent to the circle must necessary have intersection points where the circle passed
    # through its sides.
    edges = (edge_pixels > 1).astype(long)
    interior = (interior > 0).astype(long) * (1 - edges)

    return edges, interior, arr


def sample_disk(double radius, (double, double) center, double[:] axis):
    cdef double dx = axis[1] - axis[0]
    cdef Py_ssize_t N = axis.shape[0]
    cdef double start = axis[0]
    cdef double xc = center[0]
    cdef double yc = center[1]
    cdef long quadrant
    # Largest m for which axis[m] <= yc
    # Largest n for which axis[n] <= xc

    # arr = np.zeros((N, N), dtype=np.float64)
    edges, interior, arr = find_edges(radius, center, axis)

    cdef long[:, :] edges_v = edges
    cdef long[:, :] interior_v = interior
    cdef double[:, :] arr_v = arr

    # Largest m for which axis[m] <= xc
    cdef Py_ssize_t origin_row = <Py_ssize_t>cmath.ceil((yc - start) / dx)

    # Largest n for which axis[n] <= yc
    cdef Py_ssize_t origin_col = <Py_ssize_t>cmath.ceil((xc - start) / dx)

    # Slice up the edge array into four subarrays at the center of the circle
    cdef Py_ssize_t row, col, m, n, num_rows, num_cols
    cdef double dh, dv

    """
    For the pixel with index [m, n]:
        - Left edge is edges[n]
        - Right edge is edges[n + 1]
        - Top edge is edges[m + 1]
        - Bottom edge is edges[m]
    i.e. edges[n] is the left edge of disk[:, n] or the bottom edge of disk[n, :]
    """
    cdef double edge_start = axis[0] - 0.5 * dx
    cdef double [:] edge_lines = np.arange(edge_start, edge_start + (N + 1) * dx, dx,
                                           dtype=np.float64)
    cdef double[:] edge_lines_v = edge_lines
    cdef double qsum

    # Note: m, n are the local indices for the top_left subarray
    # row, col are the global indices for each pixel in self.edges
    """
    Top left:
        - Iterating left-to-right and top-to-bottom
        - Chords placed at right and bottom sides of each pixel
        - Subtract all areas above and to the left (larger m and smaller n) of current pixel
    """
    quadrant = 3

    for m in range(N - 1, origin_row - 1, -1):
        for n in range(0, origin_col):
            if edges_v[m, n]:
                # Calculate displacement of vertical and horizontal chords away from circle center
                dv = xc - edge_lines_v[n + 1]
                dh = yc - edge_lines_v[m]

                # Calculated accumulated area of all pixels above and to the left of current pixel
                qsum = 0
                for i in range(m, N):
                    for j in range(0, n + 1):
                        qsum += arr_v[i, j]

                # Update array
                arr_v[m, n] = (arr_v[m, n] + quadrant_area(radius, dh, dv, quadrant)
                                   - qsum)

    """
    Top right:
        - Iterating right-to-left and top-to-bottom
        - Chords placed at left and bottom sides of each pixel
        - Subtract all areas above and to the right (larger m and larger n) of current pixel
    """
    quadrant = 4

    for m in range(N - 1, origin_row - 1, -1):
        for n in range(N - 1, origin_col - 1, -1):
            if edges_v[m, n]:
                # Calculate displacement of vertical and horizontal chords away from circle center
                dv = xc - edge_lines_v[n]
                dh = yc - edge_lines_v[m]

                # Calculated accumulated area of all pixels above and to the right of current pixel
                qsum = 0
                for i in range(m, N):
                    for j in range(n, N):
                        qsum += arr_v[i, j]

                arr_v[m, n] = (arr_v[m, n] + quadrant_area(radius, dh, dv, quadrant)
                                   - qsum)


    """
    Bottom right:
        - Iterating right-to-left and bottom-to-top
        - Chords placed at left and top sides of each pixel
        - Subtract all areas below and to the right (smaller m and larger n) of current pixel
    """
    quadrant = 1

    for m in range(0, origin_row):
        for n in range(N - 1, origin_col - 1, -1):
            if edges_v[m, n]:
                # Calculate displacement of vertical and horizontal chords away from circle center
                dv = xc - edge_lines_v[n]
                dh = yc - edge_lines_v[m + 1]

                # Calculated accumulated area of all pixels below and to the right of current pixel
                qsum = 0
                for i in range(0, m + 1):
                    for j in range(n, N):
                        qsum += arr_v[i, j]

                arr_v[m, n] = (arr_v[m, n] + quadrant_area(radius, dh, dv, quadrant)
                                   - qsum)

    """
    Bottom left:
        - Iterating left-to-right and bottom-to-top
        - Chords placed at right and top sides of each pixel
        - Subtract all areas below and to the left (smaller m and smaller n) of current pixel
    """
    quadrant = 2

    for m in range(0, origin_row):
        for n in range(0, origin_col):
            if edges_v[m, n]:
                # Calculate displacement of vertical and horizontal chords away from circle center
                dv = xc - edge_lines_v[n + 1]
                dh = yc - edge_lines_v[m + 1]

                # Calculated accumulated area of all pixels below and to the left of current pixel
                qsum = 0
                for i in range(0, m + 1):
                    for j in range(0, n + 1):
                        qsum += arr_v[i, j]

                arr_v[m, n] = (arr_v[m, n] + quadrant_area(radius, dh, dv, quadrant)
                                   - qsum)

    arr = arr / (dx * dx) + interior
    return arr


def disk_artist(radius, center):
    """
    Get a red circle with the correct size and location, for visualization purposes.

    Returns
    -------
    plt.Circle

    """
    xc, yc = center
    return plt.Circle((xc, yc), radius, color='r', fill=False)
